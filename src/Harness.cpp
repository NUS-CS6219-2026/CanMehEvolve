/// @file
/// CanMehHarness -- the evaluation harness for patch-level search over the FIT SDK.
///
/// CanMehHarness keeps the pristine baseline and the working tree, rebuilds
/// what a patch changed, applies and generates patches, and runs every
/// measurement. CanMehSearch drives CanMehHarness over the command line and
/// never touches the build.
///
/// Every subcommand prints one JSON object on stdout. CLI11 generates the help
/// text from the option definitions in main, so no second copy of that text
/// can drift from the definitions.
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <CLI11.hpp>
#include <nlohmann/json.hpp>

#include "utilities/Portability.hpp"
#include "utilities/Process.hpp"

using utilities::libraryNameFlags;
using utilities::sharedLibrarySuffix;
using utilities::readFile;
using utilities::writeFile;
using utilities::readJson;
using utilities::runProcess;
using utilities::ProcessResult;

using nlohmann::json;

namespace {

#ifndef CANMEH_DEFAULT_SDK
#define CANMEH_DEFAULT_SDK ""
#endif
#ifndef CANMEH_DEFAULT_WORKLOAD
#define CANMEH_DEFAULT_WORKLOAD "workloads/SdkDecode.cpp"
#endif
#ifndef CANMEH_DEFAULT_BENCH
#define CANMEH_DEFAULT_BENCH "./CanMehBench"
#endif
#ifndef CANMEH_DEFAULT_DATA
#define CANMEH_DEFAULT_DATA "data"
#endif
#ifndef CANMEH_DEFAULT_CXX
#define CANMEH_DEFAULT_CXX "c++"
#endif

auto compiler = std::string{CANMEH_DEFAULT_CXX};
auto standardFlag = std::string{"-std=c++17"};
auto optimizationFlag = std::string{"-O2"};


std::vector<std::string> splitLines(const std::string& text) {
  auto lines = std::vector<std::string>{};
  auto in = std::istringstream{text};
  auto line = std::string{};
  while (std::getline(in, line)) lines.push_back(line + "\n");
  // Every line keeps its separator, because splice rebuilds a file by
  // concatenating the lines. getline cannot report whether getline stopped at a
  // separator or at the end of the text, so this function trims the separator
  // back off a last line that never had one.
  if (!lines.empty() && text.back() != '\n') lines.back().pop_back();
  return lines;
}

std::string trim(const std::string& text) {
  const auto first = size_t{text.find_first_not_of(" \t\r\n")};
  if (first == std::string::npos) return "";
  const auto last = size_t{text.find_last_not_of(" \t\r\n")};
  return text.substr(first, last - first + 1);
}

std::string readPatchOrDie(const std::filesystem::path& path);

[[noreturn]] void reportErrorAndExit(const std::string& stage, const std::string& message) {
  std::cout << json{{"ok", false}, {"stage", stage}, {"error", message}}.dump() << "\n";
  std::exit(0);
}

std::string readPatchOrDie(const std::filesystem::path& path) {
  if (path.empty()) return "";
  if (!std::filesystem::exists(path)) reportErrorAndExit("args", "no such patch file: " + path.string());
  return readFile(path);
}

std::vector<std::string> fileNamesWithExtension(const std::filesystem::path& dir, const std::string& suffix) {
  auto names = std::vector<std::string>{};
  auto errorCode = std::error_code{};
  for (const auto& entry : std::filesystem::directory_iterator(dir, errorCode))
    if (entry.path().extension() == suffix) names.push_back(entry.path().filename().string());
  std::sort(names.begin(), names.end());
  return names;
}

struct Region {
  std::string file;   // relative to the SDK root, e.g. src/fit_decode.cpp
  int start = 0;      // 1-based, inclusive
  int end = 0;        // 1-based, inclusive
  // What labelForSignature read off the signature line, or "anonymous" when
  // labelForSignature found no name.
  std::string label;
  int ordinal = 0;    // which same-label region in this file, so overloads stay distinct
};

/// True when the accumulated signature text opens a function body rather than a
/// data initializer, a class body or a control-flow block. looksLikeFunction
/// reads only the signature text, so looksLikeFunction rejects whatever it
/// cannot classify.
bool looksLikeFunction(const std::string& signature) {
  const auto text = std::string{trim(signature)};
  if (text.empty()) return false;
  if (text[0] == '#' || text[0] == '/' || text[0] == '*') return false;
  if (text.find('(') == std::string::npos) return false;
  if (text.find('=') != std::string::npos && text.find("operator") == std::string::npos)
    return false;  // a data initializer, or a defaulted declaration
  // Reject control flow, which also has parentheses followed by a brace.
  static const auto controlFlowKeywords = {"if",     "for",    "while", "switch", "catch",
                                    "return", "else",   "do",    "class",  "struct",
                                    "enum",   "union",  "namespace"};
  const auto identifierEnd = std::find_if_not(text.begin(), text.end(), [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
  });
  const auto firstWord = std::string{text.begin(), identifierEnd};
  if (std::any_of(controlFlowKeywords.begin(), controlFlowKeywords.end(),
                  [&firstWord](const char* k) { return firstWord == k; }))
    return false;
  const auto close = size_t{text.rfind(')')};
  if (close == std::string::npos) return false;
  static const auto trailingQualifiers = {"const", "noexcept", "override", "final",
                                      "throw", "&",        "&&"};
  auto words = std::istringstream{trim(text.substr(close + 1))};
  auto word = std::string{};
  while (words >> word)
    if (std::none_of(trailingQualifiers.begin(), trailingQualifiers.end(),
                     [&word](const char* q) { return word.find(q) != std::string::npos; }))
      return false;
  return true;
}

std::string labelForSignature(const std::string& signature) {
  const auto text = std::string{trim(signature)};
  const auto open = size_t{text.find('(')};
  auto label = std::string{open == std::string::npos ? text : text.substr(0, open)};
  label = trim(label);
  const auto space = size_t{label.find_last_of(" \t*&")};
  if (space != std::string::npos) label = label.substr(space + 1);
  return label.empty() ? "anonymous" : label;
}

/// regionsInFile matches braces to find each function body. regionsInFile skips
/// any construct looksLikeFunction rejects, so a region regionsInFile does emit
/// always spans a whole body and runSplice can replace it.
std::vector<Region> regionsInFile(const std::filesystem::path& root, const std::string& rel,
                                 int maxLines) {
  const auto lines = std::vector<std::string>{splitLines(readFile(root / rel))};
  auto out = std::vector<Region>{};

  auto depth = int{0};
  auto signature = std::string{};
  auto signatureStart = int{0};
  auto inBlockComment = bool{false};
  auto inFunction = bool{false};
  auto functionDepth = int{0};
  auto functionStart = int{0};
  auto functionLabel = std::string{};

  for (auto i = size_t{0}; i < lines.size(); ++i) {
    const auto& line = lines[i];
    auto code = std::string{};
    // This loop copies the line into code without its comments or string
    // literals, so the brace counter below never sees a brace inside one.
    for (auto c = size_t{0}; c < line.size(); ++c) {
      if (inBlockComment) {
        if (line[c] == '*' && c + 1 < line.size() && line[c + 1] == '/') {
          inBlockComment = false;
          ++c;
        }
        continue;
      }
      if (line[c] == '/' && c + 1 < line.size() && line[c + 1] == '*') {
        inBlockComment = true;
        ++c;
        continue;
      }
      if (line[c] == '/' && c + 1 < line.size() && line[c + 1] == '/') break;
      if (line[c] == '"' || line[c] == '\'') {
        const auto quote = char{line[c]};
        ++c;
        while (c < line.size() && line[c] != quote) {
          if (line[c] == '\\') ++c;
          ++c;
        }
        continue;
      }
      code += line[c];
    }

    // A function body does not open at depth zero: every SDK file wraps its
    // contents in `namespace fit`, and the headers nest members inside a class
    // as well. So this scanner records the depth each body opened at, rather
    // than assuming that depth is zero.
    for (char c : code) {
      if (c == '{') {
        if (!inFunction && looksLikeFunction(signature)) {
          inFunction = true;
          functionDepth = depth;
          functionStart = signatureStart;
          functionLabel = labelForSignature(signature);
        } else if (!inFunction) {
          signature.clear();
          signatureStart = 0;
        }
        ++depth;
      } else if (c == '}') {
        --depth;
        if (depth < 0) depth = 0;
        if (inFunction && depth == functionDepth) {
          const auto end = int{static_cast<int>(i) + 1};
          if (functionStart != 0 && end - functionStart + 1 <= maxLines) {
            const auto ordinal = int{static_cast<int>(
                std::count_if(out.begin(), out.end(), [&functionLabel](const Region& seen) {
                  return seen.label == functionLabel;
                }))};
            out.push_back({rel, functionStart, end, functionLabel, ordinal});
          }
          inFunction = false;
          signature.clear();
          signatureStart = 0;
        }
      }
    }

    if (!inFunction) {
      const auto t = std::string{trim(code)};
      if (t.empty()) {
        signature.clear();
        signatureStart = 0;
      } else if (code.find('{') == std::string::npos &&
                 code.find('}') == std::string::npos) {
        if (signature.empty()) signatureStart = static_cast<int>(i) + 1;
        signature += " " + t;
        if (t.back() == ';' || t.back() == ':') {
          signature.clear();
          signatureStart = 0;
        }
      }
    }
  }
  return out;
}

std::vector<Region> regionsInTree(const std::filesystem::path& root, int maxLines,
                                const std::string& onlyFile) {
  auto out = std::vector<Region>{};
  auto files = std::vector<std::string>{};
  for (const std::string& name : fileNamesWithExtension(root / "src", ".cpp")) files.push_back("src/" + name);
  for (const std::string& name : fileNamesWithExtension(root / "src", ".hpp")) files.push_back("src/" + name);
  for (const std::string& rel : files) {
    if (!onlyFile.empty() && rel.find(onlyFile) == std::string::npos) continue;
    const auto found = std::vector<Region>{regionsInFile(root, rel, maxLines)};
    out.insert(out.end(), found.begin(), found.end());
  }
  return out;
}

/// Line numbers shift as soon as a patch is applied, so `show` and `splice`
/// re-find a region in the patched text rather than trusting the offsets the
/// baseline listing recorded. Overloads share a label, which is why the match
/// has to be on the ordinal as well.
Region regionInTreeOrDie(const std::filesystem::path& root, const Region& fromBaseline,
                          int maxLines) {
  const auto current = std::vector<Region>{regionsInFile(root, fromBaseline.file, maxLines)};
  const auto match = std::find_if(current.begin(), current.end(), [&fromBaseline](const Region& c) {
    return c.label == fromBaseline.label && c.ordinal == fromBaseline.ordinal;
  });
  if (match == current.end())
    reportErrorAndExit("region", "region " + fromBaseline.label + " no longer exists under this patch");
  return *match;
}

struct WorkDirectory {
  /// Canonicalized once here, because a child process running with a different
  /// working directory resolves a relative --work differently or not at all.
  /// weakly_canonical rather than absolute, because absolute leaves `..` and
  /// symlinks in the path for the child to resolve against its own directory.
  explicit WorkDirectory(const std::filesystem::path& given) {
    auto errorCode = std::error_code{};
    const auto resolved = std::filesystem::weakly_canonical(given, errorCode);
    dir = errorCode ? given : resolved;
  }

  std::filesystem::path dir;
  std::filesystem::path baseline() const { return dir / "baseline"; }
  std::filesystem::path tree() const { return dir / "tree"; }
  std::filesystem::path objectDirectory() const { return dir / "obj"; }
  /// Each build is a shared library exporting the workload's C entry points, so
  /// CanMehBench can hold both libraries at once under RTLD_LOCAL and Google
  /// Benchmark can interleave the repetitions of the two benchmark functions.
  /// CanMehHarness links the reference library once, during init, and never
  /// rebuilds that library.
  std::filesystem::path candidateLibrary() const {
    return dir / ("libfit_cand" + std::string{sharedLibrarySuffix});
  }
  std::filesystem::path referenceLibrary() const {
    return dir / ("libfit_ref" + std::string{sharedLibrarySuffix});
  }
  /// A third library, compiled with debug line tables so a sampler can name a
  /// source line, and its own object directory. Keeping these apart from obj/
  /// is what stops the debug objects reaching libfit_cand: linkLibrary takes
  /// every .o in the directory it is handed, and unitsToRebuild recompiles only
  /// stale stems, so a -g object written into obj/ would persist and leave
  /// libfit_cand holding debug information that libfit_ref, linked once during
  /// init, would not -- a difference between the two measured libraries that no
  /// patch caused.
  std::filesystem::path samplingObjectDirectory() const { return dir / "obj-sampling"; }
  std::filesystem::path samplingLibrary() const {
    return dir / ("libfit_sampling" + std::string{sharedLibrarySuffix});
  }
  std::filesystem::path scratch() const { return dir / "scratch"; }
};

std::filesystem::path dependencyMapPath(const WorkDirectory& work) { return work.dir / "deps.txt"; }

std::map<std::string, std::set<std::string>> dependenciesByObjectStem(
    const std::filesystem::path& dependencyMap) {
  auto deps = std::map<std::string, std::set<std::string>>{};
  auto in = std::ifstream{dependencyMap};
  auto line = std::string{};
  while (std::getline(in, line)) {
    const auto colon = size_t{line.find(':')};
    if (colon == std::string::npos) continue;
    const auto stem = std::string{line.substr(0, colon)};
    auto rest = std::istringstream{line.substr(colon + 1)};
    auto dependency = std::string{};
    while (rest >> dependency) deps[stem].insert(dependency);
  }
  return deps;
}

std::vector<std::string> translationUnitStems(const std::filesystem::path& sourceDirectory) {
  auto stems = std::vector<std::string>{};
  for (const std::string& name : fileNamesWithExtension(sourceDirectory, ".cpp"))
    stems.push_back(std::filesystem::path{name}.stem().string());
  return stems;
}

void writeDependencyMap(const WorkDirectory& work) {
  auto out = std::ofstream{dependencyMapPath(work)};
  const auto sourceDirectory = std::filesystem::path{work.tree() / "src"};
  // The compiler names each dependency as an absolute path, while deps.txt
  // records paths relative to the tree, so treePrefix is what this loop strips
  // off each dependency.
  const auto treePrefix = std::string{work.tree().string() + "/"};
  for (const std::string& stem : translationUnitStems(sourceDirectory)) {
    const auto dependencyFile = std::filesystem::path{work.scratch() / "dep.out"};
    runProcess({compiler, "-MM", standardFlag, "-I" + sourceDirectory.string(),
                 (sourceDirectory / (stem + ".cpp")).string()},
                dependencyFile, work.scratch() / "dep.err", 60);
    auto flat = std::string{readFile(dependencyFile)};
    for (auto at = size_t{flat.find("\\\n")}; at != std::string::npos; at = flat.find("\\\n", at))
      flat.replace(at, 2, " ");
    std::replace(flat.begin(), flat.end(), '\n', ' ');
    out << stem << ":";
    auto words = std::istringstream{flat};
    auto word = std::string{};
    while (words >> word) {
      const auto at = size_t{word.find(treePrefix)};
      if (at == std::string::npos) continue;
      out << " " << word.substr(at + treePrefix.size());
    }
    out << "\n";
  }
}

/// Compiles each named translation unit into objectDirectory.
/// @return "" when every unit compiled, and otherwise the compiler's
///         diagnostics.
///
/// -fvisibility=hidden keeps each library's symbols and static tables to
/// itself, which is what lets two builds of one library be loaded into one
/// process. Every library CanMehBench opens is compiled here, so a change to
/// this flag set reaches every measurement at once.
std::string compileTranslationUnits(const WorkDirectory& work,
                                     const std::vector<std::string>& stems,
                                     const std::filesystem::path& objectDirectory,
                                     const std::vector<std::string>& extraFlags = {}) {
  const auto sourceDirectory = std::filesystem::path{work.tree() / "src"};
  const auto diagnosticsPath = std::filesystem::path{work.scratch() / "cc.err"};
  // runProcess calls execvp directly, so a value such as "-O3 -march=native"
  // must become two arguments before it reaches the compiler.
  auto buildArguments = std::vector<std::string>{compiler};
  auto optimizationFlags = std::istringstream{optimizationFlag};
  auto flag = std::string{};
  while (optimizationFlags >> flag) buildArguments.push_back(flag);
  buildArguments.insert(buildArguments.end(), {standardFlag, "-I" + sourceDirectory.string()});
  for (const std::string& stem : stems) {
    auto argv = buildArguments;
    for (const std::string& flag : extraFlags) argv.push_back(flag);
    argv.insert(argv.end(), {"-fPIC", "-fvisibility=hidden", "-c",
                             (sourceDirectory / (stem + ".cpp")).string(), "-o",
                             (objectDirectory / (stem + ".o")).string()});
    const auto compiled = ProcessResult{
        runProcess(argv, work.scratch() / "cc.out", diagnosticsPath, 300)};
    if (!compiled.ok()) {
      const auto diagnostics = std::string{readFile(diagnosticsPath)};
      return diagnostics.empty() ? ("the compiler " + compiled.describe()) : diagnostics;
    }
  }
  return "";
}

/// Links every object in objectDirectory into outputLibrary, which claims its
/// own path as its name. dyld registers each loaded image under the name that
/// image claims and answers a later dlopen of that name with the registered
/// image, so when the reference claims the candidate's path CanMehBench opens
/// the reference twice and never runs the candidate's code. We measured both
/// directions on Darwin 27; libraryNameFlags records a soname on Linux for the
/// same reason, and nobody has measured the Linux half.
std::string linkLibrary(const WorkDirectory& work, const std::filesystem::path& outputLibrary,
                         const std::filesystem::path& objectDirectory) {
  auto argv = std::vector<std::string>{compiler, "-shared", "-o", outputLibrary.string()};
  for (const std::string& flag : libraryNameFlags(outputLibrary.string()))
    argv.push_back(flag);
  // Every object comes from a tree/src/<stem>.cpp, so an object whose source is
  // gone is an orphan of a rename or a deletion and must not reach the link.
  // linkLibrary globs the directory rather than taking a list, and
  // unitsToRebuild only ever names stale units, so nothing else would notice:
  // renaming the workload to workloads/SdkDecode.cpp left a CanMehWorkload.o in
  // both obj/ and obj-sampling/, and the link failed on six duplicate symbols.
  // A library linked from an orphan is the worst outcome this project has,
  // because it measures code no source in the tree describes.
  for (const std::string& name : fileNamesWithExtension(objectDirectory, ".o")) {
    const auto stem = std::filesystem::path{name}.stem().string();
    if (!std::filesystem::exists(work.tree() / "src" / (stem + ".cpp"))) continue;
    argv.push_back((objectDirectory / name).string());
  }
  const auto linked = ProcessResult{
      runProcess(argv, work.scratch() / "ld.out", work.scratch() / "ld.err", 300)};
  if (linked.ok()) return "";
  const auto diagnostics = std::string{readFile(work.scratch() / "ld.err")};
  return diagnostics.empty() ? ("the linker " + linked.describe()) : diagnostics;
}

/// The files a unified diff changes, named as the `+++` headers spell them with
/// any `b/` prefix and trailing timestamp removed.
std::set<std::string> filesInPatch(const std::string& patchText) {
  auto files = std::set<std::string>{};
  for (const std::string& line : splitLines(patchText)) {
    if (line.rfind("+++ ", 0) != 0) continue;
    auto p = std::string{trim(line.substr(4))};
    const auto tab = size_t{p.find('\t')};
    if (tab != std::string::npos) p = p.substr(0, tab);
    if (p.rfind("b/", 0) == 0) p = p.substr(2);
    files.insert(p);
  }
  return files;
}

/// The files the patch now in the tree changed. restoreTreeAndApplyPatch
/// restores those files before applying the next patch, so a file one candidate
/// edited cannot stay edited for the next candidate.
std::filesystem::path touchedListPath(const WorkDirectory& work) { return work.dir / "touched.txt"; }

std::set<std::string> readTouchedList(const WorkDirectory& work) {
  auto files = std::set<std::string>{};
  for (const std::string& line : splitLines(readFile(touchedListPath(work)))) {
    const auto rel = std::string{trim(line)};
    if (!rel.empty()) files.insert(rel);
  }
  return files;
}

void writeTouchedList(const WorkDirectory& work, const std::set<std::string>& files) {
  auto text = std::string{};
  for (const std::string& rel : files) text += rel + "\n";
  writeFile(touchedListPath(work), text);
}

/// Restores the working tree to the baseline, then applies `patch_text` if any.
/// @return every file restoreTreeAndApplyPatch copied back from the baseline
///         or handed to `patch`. Copying a file back changes its modification
///         time, which is how unitsToRebuild notices that a reverted file's
///         object is stale; no caller derives a rebuild set from this value.
std::set<std::string> restoreTreeAndApplyPatch(const WorkDirectory& work, const std::string& patchText,
                                      std::string* error) {
  const auto touchedByPatch = std::set<std::string>{filesInPatch(patchText)};

  auto restore = std::set<std::string>{touchedByPatch};
  const auto previous = std::set<std::string>{readTouchedList(work)};
  restore.insert(previous.begin(), previous.end());

  for (const std::string& rel : restore) {
    const auto from = std::filesystem::path{work.baseline() / rel};
    if (!std::filesystem::exists(from)) continue;
    auto errorCode = std::error_code{};
    std::filesystem::copy_file(from, work.tree() / rel,
                               std::filesystem::copy_options::overwrite_existing, errorCode);
    if (errorCode) {
      *error = "cannot restore " + rel + ": " + errorCode.message();
      return restore;
    }
  }

  writeTouchedList(work, touchedByPatch);

  if (trim(patchText).empty()) return restore;

  const auto patchPath = std::filesystem::path{work.scratch() / "candidate.patch"};
  writeFile(patchPath, patchText);
  const auto applied = ProcessResult{
      runProcess({"patch", "-p1", "--batch", "--forward", "-i", patchPath.string()},
                  work.scratch() / "patch.out", work.scratch() / "patch.err", 60, work.tree())};
  if (!applied.ok()) {
    *error = "patch did not apply: " + readFile(work.scratch() / "patch.out") +
             readFile(work.scratch() / "patch.err");
  }
  return restore;
}

std::string unifiedDiffAgainstBaseline(const WorkDirectory& work, const std::set<std::string>& files) {
  auto patch = std::string{};
  for (const std::string& rel : files) {
    const auto out = std::filesystem::path{work.scratch() / "diff.out"};
    runProcess({"diff", "-u", "--label", "a/" + rel, "--label", "b/" + rel,
                 (work.baseline() / rel).string(), (work.tree() / rel).string()},
                out, work.scratch() / "diff.err", 60);
    patch += readFile(out);
  }
  return patch;
}

/// Which object files are out of date, decided the way make decides it: an
/// object is stale if it is missing or older than any file it depends on. A
/// record of which files each patch touched cannot be trusted here: a file one
/// candidate edited and the next restored leaves an object the compiler built
/// from the patched file, and linkLibrary would then link that object against a
/// tree that no longer holds the patch.
std::vector<std::string> unitsToRebuild(const WorkDirectory& work,
                                         const std::filesystem::path& objectDirectory) {
  const auto deps = dependenciesByObjectStem(dependencyMapPath(work));
  auto stems = std::vector<std::string>{};

  for (const auto& [stem, files] : deps) {
    auto errorCode = std::error_code{};
    const auto objectTime =
        std::filesystem::last_write_time(objectDirectory / (stem + ".o"), errorCode);
    if (errorCode) {
      stems.push_back(stem);
      continue;
    }
    for (const std::string& rel : files) {
      auto sourceErrorCode = std::error_code{};
      const auto sourceTime = std::filesystem::last_write_time(work.tree() / rel, sourceErrorCode);
      if (sourceErrorCode) continue;
      if (sourceTime > objectTime) {
        stems.push_back(stem);
        break;
      }
    }
  }
  return stems;
}

/// The upper median for an even count, which is what the permutation tests
/// resample. permutationPValue calls medianOf tens of thousands of times every
/// time `measure-patch` runs, so medianOf ranks only the element it returns rather than
/// sorting
/// the whole copy.
double medianOf(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const auto middle = values.begin() + static_cast<long>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

/// Two-sided permutation test on the difference of medians. Google Benchmark
/// interleaves the two functions' repetitions, so under the null that the two
/// libraries hold the same code every assignment of the pooled timings to the
/// two labels is as likely as the observed one. p counts the relabelings whose
/// median difference is at least as large as the observed one, with one added
/// to both the numerator and the denominator so p is never zero.
double permutationPValue(const std::vector<double>& a, const std::vector<double>& b,
                     unsigned seed = 12345, int shuffles = 20000) {
  auto pool = std::vector<double>{a};
  pool.insert(pool.end(), b.begin(), b.end());
  const auto observed = double{std::abs(medianOf(a) - medianOf(b))};

  std::mt19937 rng(seed);
  auto atLeastAsExtreme = int{0};
  for (auto s = int{0}; s < shuffles; ++s) {
    std::shuffle(pool.begin(), pool.end(), rng);
    auto x = std::vector<double>(pool.begin(), pool.begin() + a.size());
    auto y = std::vector<double>(pool.begin() + a.size(), pool.end());
    if (std::abs(medianOf(x) - medianOf(y)) >= observed) ++atLeastAsExtreme;
  }
  return static_cast<double>(atLeastAsExtreme + 1) / (shuffles + 1);
}

/// The widest disagreement between processes this threshold accepts. We
/// calibrated it against repeated measurements of code-identical libraries on
/// an idle machine, so `measure-patch` reports unstable only where the
/// processes disagree more than they did in any of those. ratio_spread does
/// not separate a bias that shifts every process the same way from no bias at
/// all: a measurement can report a tight spread and a biased ratio together.
constexpr auto maxRatioSpread = double{0.25};

/// A one-sample permutation test on paired log-ratios. If the candidate and the
/// reference held the same code, the sign of each process's log-ratio would be a
/// coin flip, so p is the share of sign assignments whose mean is at least as far
/// from zero as the observed one.
///
/// signFlipPValue enumerates every assignment when there are at most 2^20 of
/// them, which is exact and which at --repeat 10 takes about ten thousand
/// additions against the 200,000 a 20,000-draw sample takes. The exact test
/// cannot report below 2 / 2^n, because the observed assignment and its negation
/// always count. A sampled estimate can fall below that floor: a cluster run at
/// --repeat 10 reported 34/20001 = 0.0016999 against a floor of
/// 2/1024 = 0.0019531, and reported it twice over, because the fixed seed draws
/// the same assignments on every call. Above 2^20 assignments signFlipPValue
/// samples and raises the estimate to the floor.
///
/// Note what this test does not read: the size of the difference. Ten processes
/// that agree in direction give the same p whether the candidate is 0.3 percent
/// faster or 10 percent faster. decisionForSummary reads the effect size
/// separately.
constexpr auto maxEnumeratedProcesses = int{20};

double signFlipPValue(const std::vector<double>& logRatios, unsigned seed = 12345,
                      int shuffles = 20000) {
  const auto processes = int{static_cast<int>(logRatios.size())};
  if (processes == 0) return 1.0;
  const auto n = double{static_cast<double>(processes)};
  const auto sum = double{std::accumulate(logRatios.begin(), logRatios.end(), 0.0)};
  const auto observed = double{std::abs(sum / n)};

  if (processes <= maxEnumeratedProcesses) {
    const auto assignments = std::uint64_t{std::uint64_t{1} << processes};
    auto atLeastAsExtreme = std::uint64_t{0};
    for (auto bits = std::uint64_t{0}; bits < assignments; ++bits) {
      auto total = double{0.0};
      for (auto i = int{0}; i < processes; ++i)
        total += ((bits >> i) & std::uint64_t{1}) != 0 ? logRatios[i] : -logRatios[i];
      if (std::abs(total / n) >= observed) ++atLeastAsExtreme;
    }
    return static_cast<double>(atLeastAsExtreme) / static_cast<double>(assignments);
  }

  std::mt19937 rng(seed);
  auto coin = std::uniform_int_distribution<int>(0, 1);
  auto atLeastAsExtreme = int{0};
  for (auto s = int{0}; s < shuffles; ++s) {
    auto total = double{0.0};
    for (double x : logRatios) total += (coin(rng) != 0) ? x : -x;
    if (std::abs(total / n) >= observed) ++atLeastAsExtreme;
  }
  const auto sampled =
      double{static_cast<double>(atLeastAsExtreme + 1) / (shuffles + 1)};
  return std::max(sampled, std::ldexp(1.0, 1 - processes));
}

/// The per-repetition timings one CanMehBench process produced.
/// runBenchmarkProcesses sets index, swapped and report; every timing in the
/// vectors below comes out of the Google Benchmark report unchanged, and
/// timingsFromBenchmarkReport computes none of them.
struct ProcessTimings {
  int index = 0;
  bool swapped = false;
  std::filesystem::path report;
  std::vector<double> candidateNs;
  std::vector<double> referenceNs;
  std::vector<double> candidateCpuNs;
  std::vector<double> referenceCpuNs;
  std::vector<long long> candidateIterations;
  std::vector<long long> referenceIterations;
  std::vector<double> loadAverage;
};

std::filesystem::path benchmarkReportDirectory(const WorkDirectory& work) {
  return work.scratch() / "gb";
}

std::filesystem::path benchmarkReportPath(const WorkDirectory& work, int index) {
  const auto padding = std::string{index < 10 ? "0" : ""};
  return benchmarkReportDirectory(work) / ("proc-" + padding + std::to_string(index) + ".json");
}

/// The only function here that knows what a Google Benchmark report looks
/// like. Everything downstream reads ProcessTimings instead.
/// @return false when the report is unreadable or either series came back
///         with fewer than three repetitions.
bool timingsFromBenchmarkReport(const json& report, ProcessTimings* into) {
  if (report.is_discarded() || !report.contains("benchmarks")) return false;
  for (const json& entry : report["benchmarks"]) {
    if (entry.value("run_type", std::string("")) != "iteration") continue;  // skip aggregates
    const auto name = std::string{entry.value("name", std::string(""))};
    const auto wall = double{entry.value("real_time", 0.0)};
    const auto processor = double{entry.value("cpu_time", 0.0)};
    // Each real_time is a mean over this many decodes, so a reader that
    // ignores the count mistakes Google Benchmark's averaging for low noise.
    const auto iterations = static_cast<long long>(entry.value("iterations", 0));
    if (name.rfind("candidate", 0) == 0) {
      into->candidateNs.push_back(wall);
      into->candidateCpuNs.push_back(processor);
      into->candidateIterations.push_back(iterations);
    } else if (name.rfind("reference", 0) == 0) {
      into->referenceNs.push_back(wall);
      into->referenceCpuNs.push_back(processor);
      into->referenceIterations.push_back(iterations);
    }
  }
  if (report.contains("context") && report["context"].contains("load_avg"))
    for (const json& value : report["context"]["load_avg"])
      into->loadAverage.push_back(value.get<double>());
  return into->candidateNs.size() >= 3 && into->referenceNs.size() >= 3;
}

/// Runs one CanMehBench process, which holds both libraries at once and lets
/// Google Benchmark interleave the repetitions of the two benchmark
/// functions, then reads the report back. Computes no statistic.
/// @return a null object on success, or the failure to report.
json runOneBenchmarkProcess(const WorkDirectory& work, const json& meta, int repetitions,
                             bool swapSlots, ProcessTimings* into) {
  const auto bench = std::string{meta.value("bench", std::string("./CanMehBench"))};
  const auto data = std::string{meta["data"].get<std::string>()};
  const auto stdoutPath = std::filesystem::path{work.scratch() / "bench.out"};

  // The candidate goes into the --base slot on alternate processes, so load
  // order stops being a constant difference between the two measurements.
  // CanMehBench keeps each series' label with its role, so both load orders
  // report the same quantities.
  auto argv = std::vector<std::string>{
      bench, "--data", data,
      "--base", (swapSlots ? work.candidateLibrary() : work.referenceLibrary()).string(),
      "--cand", (swapSlots ? work.referenceLibrary() : work.candidateLibrary()).string(),
      "--benchmark_repetitions=" + std::to_string(repetitions),
      "--benchmark_enable_random_interleaving=true",
      "--benchmark_min_time=" + meta.value("min_time", std::string("0.25s")),
      "--benchmark_out=" + into->report.string(), "--benchmark_out_format=json"};
  if (swapSlots) argv.push_back("--swap");

  const auto benchmarkRun =
      ProcessResult{runProcess(argv, stdoutPath, work.scratch() / "bench.err", 1800)};

  if (benchmarkRun.ending == ProcessResult::Ending::TimedOut)
    return json{{"ok", false}, {"stage", "run"}, {"error", "the benchmark timed out"}};

  // A candidate whose lookup tables are too small reads out of bounds and takes
  // the benchmark process down with it, so there is no JSON to read back. Name
  // the signal rather than reporting a bare exit status: CanMehBench itself
  // only ever exits 0 or 1.
  if (benchmarkRun.ending == ProcessResult::Ending::Signaled)
    return json{{"ok", false},
                {"stage", "crash"},
                {"error", "the candidate " + benchmarkRun.describe() + ": " +
                              readFile(work.scratch() / "bench.err").substr(0, 400)}};

  // CanMehBench prints a JSON rejection on stdout and exits 0 when the candidate
  // is wrong, threaded or unloadable.
  auto rejected = readJson(stdoutPath);
  if (!rejected.is_discarded() && rejected.is_object() && rejected.contains("ok") &&
      !rejected["ok"].get<bool>()) {
    return rejected;
  }
  if (!benchmarkRun.ok())
    return json{{"ok", false},
                {"stage", "run"},
                {"error", "the benchmark " + benchmarkRun.describe() + ": " +
                              readFile(work.scratch() / "bench.err").substr(0, 800)}};

  if (!timingsFromBenchmarkReport(readJson(into->report), into))
    return json{{"ok", false},
                {"stage", "run"},
                {"error", "too few repetitions came back from the benchmark"}};
  return json{};
}

/// Runs the independent CanMehBench processes behind one measurement. Each
/// process writes its report to scratch/gb/proc-NN.json, so a reader can read
/// every timing behind a reported p-value after the run.
/// @return the failure object CanMehBench or runOneBenchmarkProcess reported,
///         or a null object when every process succeeded, with the timings in
///         into.
json runBenchmarkProcesses(const WorkDirectory& work, const json& meta, int repetitions,
                            int repeat, std::vector<ProcessTimings>* into) {
  auto errorCode = std::error_code{};
  std::filesystem::remove_all(benchmarkReportDirectory(work), errorCode);
  std::filesystem::create_directories(benchmarkReportDirectory(work), errorCode);

  const auto processes = int{std::max(1, repeat)};
  for (auto index = int{0}; index < processes; ++index) {
    auto timings = ProcessTimings{};
    timings.index = index;
    // runBenchmarkProcesses gives the candidate the --base slot on every odd
    // process, so at an even --repeat any asymmetry between the two slots
    // biases half the processes each way rather than all of them the same way,
    // which is the sign exchangeability signFlipPValue assumes. At --repeat 1
    // no process swaps and nothing is exchangeable, which is one reason 10 is
    // the smallest --repeat that can reach p < 0.01.
    timings.swapped = index % 2 == 1;
    timings.report = benchmarkReportPath(work, index);
    const auto failure =
        runOneBenchmarkProcess(work, meta, repetitions, timings.swapped, &timings);
    if (!failure.is_null()) return failure;
    into->push_back(timings);
  }
  return json{};
}

double coefficientOfVariation(const std::vector<double>& values) {
  const auto count = double{static_cast<double>(values.size())};
  const auto mean = double{std::accumulate(values.begin(), values.end(), 0.0) / count};
  const auto squares = double{std::accumulate(
      values.begin(), values.end(), 0.0,
      [mean](double acc, double x) { return acc + (x - mean) * (x - mean); })};
  return std::sqrt(squares / count) / mean;
}

/// Ending the measurement ran, as opposed to what the measurement concluded.
/// These three are facts about the processes rather than statistics derived
/// from the timings, so measure-patch reports them whether or not it reports a verdict.
json measurementShape(const std::vector<ProcessTimings>& processes) {
  auto fewestRepetitions = size_t{processes.front().candidateNs.size()};
  auto swappedCount = int{0};
  for (const ProcessTimings& one : processes) {
    fewestRepetitions = std::min(fewestRepetitions, one.candidateNs.size());
    if (one.swapped) ++swappedCount;
  }
  return json{{"processes", processes.size()},
              {"swapped_processes", swappedCount},
              {"repetitions", fewestRepetitions}};
}

/// Every statistic `measure-patch` reports, and nothing else, so a caller that does not
/// call summarizeTimings reports no statistic. That is how --no-decision
/// withholds them.
///
/// summarizeTimings summarizes one process by relabeling its repetitions and
/// several processes by sign-flipping the per-process log-ratios, because the
/// ratio varies far more from one process to the next than the repetitions
/// inside one process do. A sign-flip test on n pairs cannot report below
/// 2 / 2^n, so --repeat 10 is the smallest value that can reach p < 0.01 at
/// all.
json summarizeTimings(const std::vector<ProcessTimings>& processes) {
  auto ratios = std::vector<double>{};
  auto logRatios = std::vector<double>{};
  auto candidateMs = std::vector<double>{};
  auto referenceMs = std::vector<double>{};
  auto worstCandidateCv = double{0.0};
  auto worstReferenceCv = double{0.0};

  for (const ProcessTimings& one : processes) {
    const auto candidateMedian = double{medianOf(one.candidateNs)};
    const auto referenceMedian = double{medianOf(one.referenceNs)};
    ratios.push_back(candidateMedian / referenceMedian);
    logRatios.push_back(std::log(ratios.back()));
    candidateMs.push_back(candidateMedian / 1e6);
    referenceMs.push_back(referenceMedian / 1e6);
    worstCandidateCv = std::max(worstCandidateCv, coefficientOfVariation(one.candidateNs));
    worstReferenceCv = std::max(worstReferenceCv, coefficientOfVariation(one.referenceNs));
  }

  const auto onlyOneProcess = bool{processes.size() == 1};
  const auto ratio = double{onlyOneProcess ? ratios.front() : medianOf(ratios)};
  const auto [lowest, highest] = std::minmax_element(ratios.begin(), ratios.end());
  return json{{"score_ms", onlyOneProcess ? candidateMs.front() : medianOf(candidateMs)},
              {"reference_ms", onlyOneProcess ? referenceMs.front() : medianOf(referenceMs)},
              {"ratio", ratio},
              {"speedup", 1.0 / ratio},
              {"p_value", onlyOneProcess ? permutationPValue(processes.front().candidateNs,
                                                     processes.front().referenceNs)
                                  : signFlipPValue(logRatios)},
              {"candidate_cv", worstCandidateCv},
              {"reference_cv", worstReferenceCv},
              {"ratio_spread", *highest - *lowest}};
}

json timingsJson(const std::vector<ProcessTimings>& processes, const json& meta,
                  int repetitions) {
  auto array = json::array();
  auto ragged = bool{false};
  for (const ProcessTimings& one : processes) {
    if (one.candidateNs.size() != processes.front().candidateNs.size() ||
        one.referenceNs.size() != one.candidateNs.size())
      ragged = true;
    array.push_back(json{{"index", one.index},
                         {"swapped", one.swapped},
                         {"report", one.report.string()},
                         {"load_avg", one.loadAverage},
                         {"candidate_ns", one.candidateNs},
                         {"reference_ns", one.referenceNs},
                         {"candidate_cpu_ns", one.candidateCpuNs},
                         {"reference_cpu_ns", one.referenceCpuNs},
                         {"candidate_iterations", one.candidateIterations},
                         {"reference_iterations", one.referenceIterations}});
  }
  return json{{"unit", "ns"},
              {"clock", "real_time"},
              {"repetitions_requested", repetitions},
              {"min_time", meta.value("min_time", std::string("0.25s"))},
              {"ragged", ragged},
              {"process", array}};
}

/// What `measure-patch`, `measure-pair` and `init` call. Returns the failure object as it
/// came back when any process failed, and otherwise the timings with the
/// shape of the run beside them.
///
/// withStatistics false adds no statistic at all, which is how --no-decision
/// withholds them: nothing is computed and then erased. It attaches no
/// decision either way, because `init` reads the statistics and reaches no
/// verdict from them.
json measureAcrossProcesses(const WorkDirectory& work, const json& meta, int repetitions,
                             int repeat, bool withStatistics) {
  auto processes = std::vector<ProcessTimings>{};
  // Not json{...}: braces around a json value make an array holding that
  // value, so the project's usual `const auto x = Type{expr}` spelling cannot
  // be used here.
  const auto failure = runBenchmarkProcesses(work, meta, repetitions, repeat, &processes);
  if (!failure.is_null()) return failure;

  auto out = measurementShape(processes);
  out["timings"] = timingsJson(processes, meta, repetitions);
  out["ok"] = true;
  if (withStatistics) out.update(summarizeTimings(processes));
  return out;
}

/// At --repeat 1 the within-process coefficient of variation is the only noise
/// estimate there is. Over processes, candidate_cv is the worst of N and no
/// longer means what 0.04 was calibrated for, so what matters there is whether
/// the processes agree with each other.
json decisionForSummary(const json& summary, double minEffect) {
  const auto processes = summary.value("processes", 1);
  const auto ratio = double{summary.value("ratio", 1.0)};
  const auto p = double{summary.value("p_value", 1.0)};  const auto significant = bool{ratio < 1.0 && p < 0.01};
  return json{{"significant", significant},
              {"significant_at_min_effect",
               significant && (1.0 - ratio) >= minEffect},
              {"min_effect", minEffect},
              {"unstable", processes > 1
                                ? summary.value("ratio_spread", 0.0) > maxRatioSpread
                                : (summary.value("candidate_cv", 0.0) > 0.04 ||
                                   summary.value("reference_cv", 0.0) > 0.04)}};
}

/// What measure-patch and measure-pair both accept. Each override falls back to meta.json
/// when absent, so a work directory decides how it is measured and a caller
/// overrides only what that caller means to change.
struct MeasurementOptions {
  std::optional<int> repetitions;
  std::optional<int> repeat;
  std::optional<std::string> minTime;
  /// Withholds every statistic and the decision, leaving the raw timings. A
  /// caller that can read `significant` has not written a measurement.
  bool noDecision = false;  /// The smallest ratio improvement `significant_at_min_effect` requires, as a
  /// fraction. `significant` reads the p-value and nothing else, and
  /// signFlipPValue reads only whether the processes agree in direction, so
  /// `significant` alone says nothing about magnitude: ten processes that agree
  /// in direction report the same p-value for a fraction of one percent as for
  /// ten percent. Phase 0 of README.md has you measure the smallest effect your
  /// own rule resolves, and --min-effect is where you apply that number.
  ///
  /// `measure-patch` computes both fields from the pair it measured, which for a
  /// spliced patch is the whole accumulated patch against the reference.
  /// `measure-patch` cannot report what one generation's own rewrite contributed,
  /// because `measure-patch` never reads the parent; `CanMehSearch` reports that as
  /// `improved_on_parent`.
  double minEffect = 0.02;
};

/// What `measure-patch` and `measure-pair` both report, so the two subcommands cannot drift
/// apart on what --no-decision means.
json measuredResult(const WorkDirectory& work, const json& meta, int repetitions, int repeat,
                     bool withDecision, double minEffect) {
  auto result = measureAcrossProcesses(work, meta, repetitions, repeat, withDecision);
  if (withDecision && result.value("ok", false))
    result.update(decisionForSummary(result, minEffect));
  return result;
}

/// Declared once for both subcommands, so measure-patch and measure-pair cannot come to
/// spell the same four options differently.
void addMeasurementOptions(CLI::App* command, MeasurementOptions* into) {
  command->add_option("--repetitions", into->repetitions,
                      "repetitions inside one process; meta.json decides when absent");
  command->add_option("--repeat", into->repeat,
                      "independent processes; meta.json decides when absent");
  command->add_option("--min-time", into->minTime,
                      "seconds per repetition; meta.json decides when absent");  command->add_flag("--no-decision", into->noDecision,
                    "print the timings and no statistic, so the caller decides");
  command->add_option("--min-effect", into->minEffect,
                      "smallest improvement `significant_at_min_effect` requires, "
                      "as a fraction of the reference's time")
      ->capture_default_str();
}

struct InitOptions {
  std::string work;
  std::string sdk = CANMEH_DEFAULT_SDK;
  std::string workload = CANMEH_DEFAULT_WORKLOAD;
  std::string bench = CANMEH_DEFAULT_BENCH;
  std::string data = CANMEH_DEFAULT_DATA;
  std::string minTime = "0.25s";
  int repetitions = 9;
  int repeat = 1;
  int nullRepetitions = 9;
};

struct RegionsOptions {
  std::string work;
  int maxLines = 250;
  std::string onlyFile;
};

struct ShowOptions {
  std::string work;
  int region = -1;
  int maxLines = 250;
  std::string patch;
};

struct SpliceOptions {
  std::string work;
  int region = -1;
  int maxLines = 250;
  std::string patch;
  std::string replacement;
  std::string out;
  std::string includes;
};

/// repetitions and repeat fall back to meta.json rather than to a constant, so
/// a measure-patch inherits what init calibrated with unless told otherwise. Unset is
/// therefore different from any value a flag could supply.
struct MeasurePatchOptions {
  std::string work;
  std::string patch;
  MeasurementOptions measurement;
  bool compilerGiven = false;
  bool standardGiven = false;
  bool optimizationGiven = false;
};

/// `sampling-library` builds libfit_sampling with debug line tables, so a
/// sampler can name a source line and a caller can map that line into a
/// region. It never touches the two libraries a measurement compares.
struct SamplingLibraryOptions {
  std::string work;
  bool compilerGiven = false;
  bool standardGiven = false;
  bool optimizationGiven = false;
};

struct MeasurePairOptions {
  std::string work;
  MeasurementOptions measurement;
};

void runInit(const InitOptions& opt) {
  const auto sdk = opt.sdk;
  const auto workload = opt.workload;
  const auto bench = opt.bench;
  const auto data = opt.data;
  const auto minTime = opt.minTime;
  const auto repetitions = opt.repetitions;
  const auto repeat = opt.repeat;
  auto work = WorkDirectory{opt.work};
  if (sdk.empty()) reportErrorAndExit("args", "init needs --sdk: this build compiled in no default");

  auto errorCode = std::error_code{};
  // `CanMehHarness init` re-copies the whole tree, so every object has to come
  // from that copy. Removing both object directories is what stops an object
  // whose source was renamed or deleted from surviving, because linkLibrary
  // globs a directory and unitsToRebuild only ever names stale units: renaming
  // the workload to workloads/SdkDecode.cpp left a CanMehWorkload.o in each,
  // and the baseline link failed on six duplicate symbols rather than measuring
  // code no source in the tree describes. obj-sampling/ matters for the same
  // reason even though `CanMehHarness sampling-library` fills it later:
  // unitsToRebuild would leave an orphan there untouched.
  std::filesystem::remove_all(work.baseline(), errorCode);
  std::filesystem::remove_all(work.tree(), errorCode);
  std::filesystem::remove_all(work.objectDirectory(), errorCode);
  std::filesystem::remove_all(work.samplingObjectDirectory(), errorCode);
  for (const std::filesystem::path& directory : {work.dir, work.objectDirectory(), work.scratch()})
    std::filesystem::create_directories(directory, errorCode);
  for (const std::filesystem::path& into : {work.baseline(), work.tree()}) {
    std::filesystem::copy(sdk, into, std::filesystem::copy_options::recursive, errorCode);
    if (errorCode) reportErrorAndExit("init", "cannot copy the SDK to " + into.string() + ": " + errorCode.message());
  }

  // The workload becomes one more translation unit of the library, so the same
  // rebuild and dependency tracking covers the workload as covers every SDK
  // file. init writes the workload into the baseline as well, so a diff never
  // shows the workload as an addition. The workload includes the portability
  // header, so init copies that header too, reading it from src/utilities/
  // beside the workload's own directory rather than from a separate flag, and
  // writing it to the same relative path so the include resolves under
  // -I tree/src.
  //
  // Keeping the header in utilities/ also keeps it out of the region index:
  // regionsInTree lists tree/src without descending, so no patch can reach
  // anything in a subdirectory.
  //
  // parent_path twice, because the workload is src/workloads/SdkDecode.cpp and
  // the header is src/utilities/Portability.hpp.
  const auto beside =
      std::filesystem::path{workload}.parent_path().parent_path();
  const auto sources = std::vector<std::pair<std::filesystem::path, std::string>>{
      {workload, std::filesystem::path{workload}.filename().string()},
      {beside / "utilities" / "Portability.hpp", "utilities/Portability.hpp"}};
  for (const auto& [source, relative] : sources) {
    const auto text = std::string{readFile(source)};
    if (text.empty()) reportErrorAndExit("args", "cannot read " + source.string());
    for (const std::filesystem::path& root : {work.baseline(), work.tree()}) {
      auto errorCode = std::error_code{};
      std::filesystem::create_directories((root / "src" / relative).parent_path(), errorCode);
      writeFile(root / "src" / relative, text);
    }
  }

  writeDependencyMap(work);

  const auto stems = std::vector<std::string>{translationUnitStems(work.tree() / "src")};
  const auto compileError = std::string{compileTranslationUnits(work, stems, work.objectDirectory())};
  if (!compileError.empty()) reportErrorAndExit("build", "the baseline does not compile: " + compileError.substr(0, 2000));
  const auto linkError = std::string{linkLibrary(work, work.candidateLibrary(), work.objectDirectory())};
  if (!linkError.empty()) reportErrorAndExit("build", "the baseline does not link: " + linkError.substr(0, 2000));

  // The reference library is this same build linked a second time. Every later
  // measure-patch times the candidate against the reference library inside one
  // benchmark process, so no comparison depends on what the machine was doing
  // when init ran.
  //
  // Linking the reference a second time, rather than copying the candidate file
  // aside, is what keeps the two libraries distinct on macOS: a copy would hold
  // the candidate's install name, dlopen would then return the reference image
  // for every candidate load, and every candidate would be measured and
  // checksummed as the reference.
  const auto referenceLinkError = std::string{linkLibrary(work, work.referenceLibrary(), work.objectDirectory())};
  if (!referenceLinkError.empty())
    reportErrorAndExit("build", "the reference does not link: " + referenceLinkError.substr(0, 2000));

  const auto nullRepetitions = opt.nullRepetitions;
  auto meta = json{{"data", data},
               {"cxx", compiler},
               {"cxxstd", standardFlag},
               {"opt", optimizationFlag},
               {"bench", bench},
               {"min_time", minTime},
               {"repetitions", repetitions},
               {"repeat", repeat},
               {"sdk", sdk}};

  // Null run: the reference library against an identical build of the same
  // objects. Whatever the null run reports is what this machine contributes by
  // itself, which is the one check that the measurement path is honest before
  // the harness measures any candidate with it.
  auto base = measureAcrossProcesses(work, meta, nullRepetitions, repeat, true);
  if (!base.value("ok", false))
    reportErrorAndExit("oracle", "the null measurement failed: " + base.dump().substr(0, 400));

  meta["baseline_ms"] = base["reference_ms"];
  meta["null_ratio"] = base["ratio"];
  meta["null_p"] = base["p_value"];
  meta["reference_cv"] = base["reference_cv"];
  writeFile(work.dir / "meta.json", meta.dump(2));
  writeTouchedList(work, {});

  // `init` reads the size of the difference and how noisy the machine is,
  // never p: under the null p is uniform, so refusing on p refuses good
  // machines at random, and raising --repeat makes that worse.
  const auto nullRatio = double{base.value("ratio", 1.0)};
  const auto nullP = double{base.value("p_value", 1.0)};
  const auto nullCv = double{base.value("reference_cv", 0.0)};
  const auto nullSpread = double{base.value("ratio_spread", 0.0)};
  const auto drift = double{std::abs(1.0 - nullRatio)};
  // init refuses the work directory on three measured conditions: the two
  // builds differ by enough to be mistaken for a result, the machine is too
  // noisy within a process, or the processes do not agree with each other.
  const auto nullOk = bool{drift < 0.02 && nullCv < 0.25 && nullSpread <= maxRatioSpread};

  auto outJson = json{{"ok", true},
                   {"baseline_ms", base["reference_ms"]},
                   {"null_ratio", nullRatio},
                   {"null_p", nullP},
                   {"null_spread", nullSpread},
                   {"null_ok", nullOk},
                   {"reference_cv", base["reference_cv"]},
                   {"units", stems.size()}};
  if (!nullOk) {
    outJson["warning"] =        "CanMehBench did not report a clean null run: two libraries linked from the "
        "same objects measured as a ratio of " +
        std::to_string(nullRatio) + ", a within-process cv of " +
        std::to_string(nullCv) + " and a between-process spread of " +
        std::to_string(nullSpread) +
        ". init reports null_ok false when the drift reaches 2 percent, the cv "
        "reaches 0.25, or the spread exceeds " +
        std::to_string(maxRatioSpread) +
        ". Run init again on a quiet machine, and read no result from this work "
        "directory until init reports null_ok true.";
  } else if (nullP < 0.01 && drift > 0.005) {
    // A small p beside a ratio far from 1.000 means a bias that shifts every
    // process the same way, so init prints it even when the drift stays under 2
    // percent and null_ok is true.
    outJson["note"] =
        "CanMehBench measured a consistent " + std::to_string(drift * 100.0) +
        " percent difference between two libraries linked from the same objects "
        "(p=" + std::to_string(nullP) +
        "), so this is a systematic bias and not contention. A larger --repeat "
        "makes a systematic bias more convincing rather than smaller. Find the "
        "asymmetry or measure on another machine before reading any result below "
        "about " + std::to_string(drift * 200.0) + " percent from this work "
        "directory.";
  }
  std::cout << outJson.dump() << "\n";
}

void runRegions(const RegionsOptions& opt) {
  auto work = WorkDirectory{opt.work};
  const auto maxLines = opt.maxLines;
  const auto only = opt.onlyFile;
  // Always index the complete list. --file filters what is shown, never how
  // regions are numbered: splice and show resolve an id against the unfiltered
  // list, so a filtered numbering would quietly edit a different function.
  const auto regions = std::vector<Region>{regionsInTree(work.baseline(), maxLines, "")};

  auto list = json::array();
  auto shown = size_t{0};
  for (auto i = size_t{0}; i < regions.size(); ++i) {
    if (!only.empty() && regions[i].file.find(only) == std::string::npos) continue;
    ++shown;
    list.push_back({{"id", i},
                    {"file", regions[i].file},
                    {"start", regions[i].start},
                    {"end", regions[i].end},
                    {"lines", regions[i].end - regions[i].start + 1},
                    {"label", regions[i].label}, {"ordinal", regions[i].ordinal}});
  }
  std::cout << json{{"ok", true}, {"count", shown}, {"total", regions.size()},
                    {"regions", list}}
                   .dump()
            << "\n";
}

/// Reads from the current tree state, baseline plus an optional patch, so the
/// model sees what it is actually editing.
void runShow(const ShowOptions& opt) {
  auto work = WorkDirectory{opt.work};
  const auto id = opt.region;
  const auto maxLines = opt.maxLines;
  const auto patchFile = opt.patch;

  const auto regions = std::vector<Region>{regionsInTree(work.baseline(), maxLines, "")};
  if (id < 0 || id >= static_cast<int>(regions.size())) reportErrorAndExit("args", "no such region");
  const auto& r = regions[id];

  auto err = std::string{};
  restoreTreeAndApplyPatch(work, readPatchOrDie(patchFile), &err);
  if (!err.empty()) reportErrorAndExit("patch", err.substr(0, 1000));

  const auto match = Region{regionInTreeOrDie(work.tree(), r, maxLines)};

  const auto lines = std::vector<std::string>{splitLines(readFile(work.tree() / r.file))};
  auto text = std::string{};
  for (auto i = int{match.start}; i <= match.end && i <= static_cast<int>(lines.size()); ++i)
    text += lines[i - 1];

  std::cout << json{{"ok", true},
                    {"file", r.file},
                    {"label", r.label},
                    {"start", match.start},
                    {"end", match.end},
                    {"text", text}}
                   .dump()
            << "\n";
}

void runSplice(const SpliceOptions& opt) {
  auto work = WorkDirectory{opt.work};
  const auto id = opt.region;
  const auto maxLines = opt.maxLines;
  const auto basePatch = opt.patch;
  const auto replacementFile = opt.replacement;
  const auto outFile = opt.out;

  const auto regions = std::vector<Region>{regionsInTree(work.baseline(), maxLines, "")};
  if (id < 0 || id >= static_cast<int>(regions.size())) reportErrorAndExit("args", "no such region");
  const auto& r = regions[id];

  auto err = std::string{};
  auto touched = std::set<std::string>{restoreTreeAndApplyPatch(work, readPatchOrDie(basePatch), &err)};
  if (!err.empty()) reportErrorAndExit("patch", err.substr(0, 1000));

  const auto match = Region{regionInTreeOrDie(work.tree(), r, maxLines)};

  auto replacement = std::string{readFile(replacementFile)};
  if (!replacement.empty() && replacement.back() != '\n') replacement += "\n";

  const auto lines = std::vector<std::string>{splitLines(readFile(work.tree() / r.file))};
  auto rebuilt = std::string{};
  for (auto i = int{1}; i < match.start; ++i) rebuilt += lines[i - 1];
  rebuilt += replacement;
  for (auto i = static_cast<size_t>(match.end); i < lines.size(); ++i) rebuilt += lines[i];
  // runSplice copies the replacement text in unchanged, so an #include inside
  // it lands wherever the region was rather than in the include block. The
  // --includes file is the path that works: runSplice inserts those lines after
  // the last #include already in the file, and skips a line the file already
  // contains.
  const auto includesFile = opt.includes;
  auto includesAdded = int{0};
  if (!includesFile.empty()) {
    const auto rebuiltLines = std::vector<std::string>{splitLines(rebuilt)};
    auto lastInclude = int{-1};
    for (auto i = size_t{0}; i < rebuiltLines.size(); ++i)
      if (trim(rebuiltLines[i]).rfind("#include", 0) == 0) lastInclude = static_cast<int>(i);

    auto additions = std::string{};
    for (const std::string& raw : splitLines(readFile(includesFile))) {
      const auto line = std::string{trim(raw)};
      if (line.rfind("#include", 0) != 0) continue;
      if (rebuilt.find(line) != std::string::npos) continue;
      additions += line + "\n";
      ++includesAdded;
    }
    if (!additions.empty()) {
      auto merged = std::string{};
      for (auto i = size_t{0}; i < rebuiltLines.size(); ++i) {
        merged += rebuiltLines[i];
        if (static_cast<int>(i) == lastInclude) merged += additions;
      }
      if (lastInclude < 0) merged = additions + merged;
      rebuilt = merged;
    }
  }

  writeFile(work.tree() / r.file, rebuilt);

  touched.insert(r.file);
  const auto patch = std::string{unifiedDiffAgainstBaseline(work, touched)};
  writeFile(outFile, patch);

  // Record only the files the new patch actually changes, not everything that
  // was restored on the way here, so the next rebuild stays incremental.
  const auto changed = std::set<std::string>{filesInPatch(patch)};
  writeTouchedList(work, changed);

  std::cout << json{{"ok", true}, {"patch", outFile}, {"bytes", patch.size()},
                    {"files", changed.size()}, {"includes_added", includesAdded}}
                   .dump()
            << "\n";
}

void runMeasurePatch(const MeasurePatchOptions& opt) {
  auto work = WorkDirectory{opt.work};
  const auto patchFile = opt.patch;
  auto meta = readJson(work.dir / "meta.json");
  if (meta.is_discarded()) reportErrorAndExit("args", "run init first: no meta.json in the work directory");
  // Candidates must be compiled by the same compiler that built the reference
  // library, or the comparison is between toolchains rather than patches.
  if (!opt.compilerGiven) compiler = meta.value("cxx", compiler);
  // Same reason as the compiler: a candidate compiled at a different
  // optimization level from the reference library makes the measurement a
  // comparison between toolchains rather than between patches.
  if (!opt.standardGiven) standardFlag = meta.value("cxxstd", standardFlag);
  if (!opt.optimizationGiven) optimizationFlag = meta.value("opt", optimizationFlag);

  const auto start = std::chrono::steady_clock::now();
  auto err = std::string{};
  restoreTreeAndApplyPatch(work, readPatchOrDie(patchFile), &err);
  if (!err.empty()) {
    std::cout << json{{"ok", false}, {"stage", "patch"}, {"error", err.substr(0, 1500)}}.dump()
              << "\n";
    return;
  }

  const auto stems = std::vector<std::string>{unitsToRebuild(work, work.objectDirectory())};
  const auto compileError = std::string{compileTranslationUnits(work, stems, work.objectDirectory())};
  if (!compileError.empty()) {
    std::cout << json{{"ok", false},
                      {"stage", "compile"},
                      {"rebuilt", stems.size()},
                      {"error", compileError.substr(0, 3000)}}
                     .dump()
              << "\n";
    return;
  }
  const auto linkError = std::string{linkLibrary(work, work.candidateLibrary(), work.objectDirectory())};
  if (!linkError.empty()) {
    std::cout << json{{"ok", false}, {"stage", "link"}, {"error", linkError.substr(0, 2000)}}.dump()
              << "\n";
    return;
  }
  const auto buildSeconds = double{std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()};

  const auto repetitions = opt.measurement.repetitions.value_or(meta.value("repetitions", 9));
  const auto repeat = opt.measurement.repeat.value_or(meta.value("repeat", 1));
  // decisionForSummary reports significant only when the ratio is below 1.0 and
  // p is below 0.01. summarizeTimings computes p by relabeling repetitions at
  // --repeat 1 and by sign-flipping the per-process log-ratios above that, so
  // no threshold on the percentage difference reaches the verdict.
  auto result = measuredResult(work, meta, repetitions, repeat, !opt.measurement.noDecision,
                                 opt.measurement.minEffect);
  result["rebuilt"] = stems.size();
  if (result.value("ok", false)) {
    result["build_s"] = buildSeconds;
    result["baseline_ms"] = meta.value("baseline_ms", 0.0);
  }
  std::cout << result.dump() << "\n";
}

/// Compiles the baseline tree with debug line tables into its own object
/// directory and links it into its own library.
///
/// It restores the tree to the baseline first. A sampler reports the paths and
/// lines the compiler recorded, which come from `tree()`, while `regions`
/// indexes `baseline()`; the two agree on line numbers only while no patch is
/// applied.
///
/// Clang documents -g as not changing what it emits at -O2. We checked only
/// that libfit_sampling decodes to the same checksum as libfit_cand; nobody has
/// compared the two libraries' __text sections.
///
/// On Apple the compiler leaves the DWARF in each object file and the linker
/// records one OSO entry per object, holding that object's path and build time,
/// so sample and atos need those objects still in place. dsymutil copies the
/// DWARF into a .dSYM beside the library, which is what lets a later
/// sampling-library run recompile obj-sampling without making the previous
/// library's lines unresolvable.
void runSamplingLibrary(const SamplingLibraryOptions& opt) {
  auto work = WorkDirectory{opt.work};
  auto meta = readJson(work.dir / "meta.json");
  if (meta.is_discarded())
    reportErrorAndExit("args", "no meta.json in " + work.dir.string());
  if (!opt.compilerGiven) compiler = meta.value("cxx", compiler);
  // Same reason as the compiler: a candidate compiled at a different
  // optimization level from the reference library makes the measurement a
  // comparison between toolchains rather than between patches.
  if (!opt.standardGiven) standardFlag = meta.value("cxxstd", standardFlag);
  if (!opt.optimizationGiven) optimizationFlag = meta.value("opt", optimizationFlag);

  auto errorCode = std::error_code{};
  std::filesystem::create_directories(work.samplingObjectDirectory(), errorCode);
  auto restoreError = std::string{};
  restoreTreeAndApplyPatch(work, "", &restoreError);
  if (!restoreError.empty())
    reportErrorAndExit("patch", "cannot restore the tree to the baseline: " + restoreError);

  const auto stems =
      std::vector<std::string>{unitsToRebuild(work, work.samplingObjectDirectory())};
  const auto compileError = std::string{compileTranslationUnits(
      work, stems, work.samplingObjectDirectory(), {"-g"})};
  if (!compileError.empty())
    reportErrorAndExit("build", "the sampling library does not compile: " +
                                    compileError.substr(0, 2000));
  const auto linkError = std::string{
      linkLibrary(work, work.samplingLibrary(), work.samplingObjectDirectory())};
  if (!linkError.empty())
    reportErrorAndExit("build", "the sampling library does not link: " +
                                    linkError.substr(0, 2000));

  auto debugSymbols = std::filesystem::path{};
#ifdef __APPLE__
  const auto collected = ProcessResult{runProcess({"dsymutil", work.samplingLibrary().string()},
                                                   work.scratch() / "dsym.out",
                                                   work.scratch() / "dsym.err", 300)};
  if (!collected.ok())
    reportErrorAndExit("build", "dsymutil " + collected.describe() + ": " +
                                    readFile(work.scratch() / "dsym.err").substr(0, 400));
  debugSymbols = work.samplingLibrary().string() + ".dSYM/Contents/Resources/DWARF/" +
                 work.samplingLibrary().filename().string();
#else
  debugSymbols = work.samplingLibrary();
#endif

  // Say whether the debug information actually arrived, rather than reporting
  // a successful link and leaving a caller to discover that no line resolves.
  const auto symbolSize = std::uintmax_t{std::filesystem::file_size(debugSymbols, errorCode)};
  const auto resolvable = bool{!errorCode && symbolSize > 0};

  std::cout << json{{"ok", true},
                    {"library", work.samplingLibrary().string()},
                    {"symbols", debugSymbols.string()},
                    {"symbols_bytes", resolvable ? symbolSize : 0},
                    {"resolvable", resolvable},
                    {"rebuilt", stems.size()}}
                   .dump()
            << "\n";
}

/// Times the two libraries already in a work directory. runMeasurePair applies no
/// patch and compiles nothing, so runMeasurePair also measures a work directory
/// CMake wrote for a nullRepetitions pair, where no SDK tree exists.
void runMeasurePair(const MeasurePairOptions& opt) {
  auto work = WorkDirectory{opt.work};
  auto meta = readJson(work.dir / "meta.json");
  if (meta.is_discarded())
    reportErrorAndExit("args", "no meta.json in " + work.dir.string());
  if (!meta.contains("data"))
    reportErrorAndExit("args", "meta.json names no data directory");

  const auto repetitions = int{opt.measurement.repetitions.value_or(meta.value("repetitions", 9))};
  const auto repeat = int{opt.measurement.repeat.value_or(meta.value("repeat", 1))};
  if (opt.measurement.minTime) meta["min_time"] = *opt.measurement.minTime;
  std::cout << measuredResult(work, meta, repetitions, repeat, !opt.measurement.noDecision,
                                 opt.measurement.minEffect).dump()
            << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  auto app = CLI::App{"Applies a candidate patch to the FIT SDK, builds the patched tree "
                      "into a shared library and measures that library against the "
                      "reference."};
  app.require_subcommand(1);

  /// Every subcommand that compiles anything reads these three, so main
  /// declares them on the program rather than on one subcommand.
  auto compilerOverride = std::optional<std::string>{};
  app.add_option("--cxx", compilerOverride, "compiler for every build in this run");
  auto standardOverride = std::optional<std::string>{};
  auto optimizationOverride = std::optional<std::string>{};
  app.add_option("--cxxstd", standardOverride, "language standard flag (default -std=c++17)");
  app.add_option("--opt", optimizationOverride, "whitespace-separated optimization flags (default -O2)");

  auto initOptions = InitOptions{};
  auto* init = app.add_subcommand("init", "Copy the SDK, link the reference library and run a null measurement");
  init->add_option("--work", initOptions.work, "work directory to create")->required();
  init->add_option("--sdk", initOptions.sdk, "SDK tree to copy")->capture_default_str();
  init->add_option("--workload", initOptions.workload, "workload translation unit")
      ->capture_default_str();
  init->add_option("--bench", initOptions.bench, "CanMehBench binary")->capture_default_str();
  init->add_option("--data", initOptions.data, "directory of .fit files")->capture_default_str();
  init->add_option("--min-time", initOptions.minTime, "seconds per repetition")
      ->capture_default_str();
  init->add_option("--repetitions", initOptions.repetitions, "repetitions inside one process")
      ->capture_default_str();
  init->add_option("--repeat", initOptions.repeat, "independent processes per candidate")
      ->capture_default_str();
  init->add_option("--null-repetitions", initOptions.nullRepetitions,
                   "repetitions for the null run")
      ->capture_default_str();

  auto regionsOptions = RegionsOptions{};
  auto* regions = app.add_subcommand("regions", "List the function-body regions of the baseline");
  regions->add_option("--work", regionsOptions.work, "work directory")->required();
  regions->add_option("--max-lines", regionsOptions.maxLines, "largest region to index")
      ->capture_default_str();
  regions->add_option("--file", regionsOptions.onlyFile,
                      "show only this file; ids stay those of the full list");

  auto showOptions = ShowOptions{};
  auto* show = app.add_subcommand("show", "Print the text of one region");
  show->add_option("--work", showOptions.work, "work directory")->required();
  show->add_option("--region", showOptions.region, "region id from the full list");
  show->add_option("--max-lines", showOptions.maxLines, "largest region to index")
      ->capture_default_str();
  show->add_option("--patch", showOptions.patch, "apply this patch before reading");

  auto spliceOptions = SpliceOptions{};
  auto* splice = app.add_subcommand("splice", "Replace one region and write the resulting patch");
  splice->add_option("--work", spliceOptions.work, "work directory")->required();
  splice->add_option("--region", spliceOptions.region, "region id from the full list");
  splice->add_option("--max-lines", spliceOptions.maxLines, "largest region to index")
      ->capture_default_str();
  splice->add_option("--patch", spliceOptions.patch, "patch the replacement builds on");
  splice->add_option("--replacement", spliceOptions.replacement, "file holding the new text")
      ->required();
  splice->add_option("--out", spliceOptions.out, "patch file to write")->required();
  splice->add_option("--includes", spliceOptions.includes, "file of additional include lines");

  auto measurePatchOptions = MeasurePatchOptions{};
  auto* measurePatch = app.add_subcommand(
      "measure-patch", "Build one candidate patch and time it against the reference library");
  measurePatch->add_option("--work", measurePatchOptions.work, "work directory")->required();
  measurePatch->add_option("--patch", measurePatchOptions.patch, "candidate patch");
  addMeasurementOptions(measurePatch, &measurePatchOptions.measurement);

  auto samplingOptions = SamplingLibraryOptions{};
  auto* samplingLibrary = app.add_subcommand(
      "sampling-library", "Build libfit_sampling with debug line tables, which a sampler needs to name a source line");
  samplingLibrary->add_option("--work", samplingOptions.work, "work directory")->required();

  auto measurePairOptions = MeasurePairOptions{};
  auto* measurePair = app.add_subcommand(
      "measure-pair", "Time the two libraries already in a work directory, applying no patch");
  measurePair->add_option("--work", measurePairOptions.work, "work directory")->required();
  addMeasurementOptions(measurePair, &measurePairOptions.measurement);

  CLI11_PARSE(app, argc, argv);

  if (compilerOverride) compiler = *compilerOverride;
  if (standardOverride) standardFlag = *standardOverride;
  if (optimizationOverride) optimizationFlag = *optimizationOverride;
  measurePatchOptions.compilerGiven = compilerOverride.has_value();
  measurePatchOptions.standardGiven = standardOverride.has_value();
  measurePatchOptions.optimizationGiven = optimizationOverride.has_value();
  samplingOptions.compilerGiven = compilerOverride.has_value();
  samplingOptions.standardGiven = standardOverride.has_value();
  samplingOptions.optimizationGiven = optimizationOverride.has_value();

  if (init->parsed()) runInit(initOptions);
  else if (regions->parsed()) runRegions(regionsOptions);
  else if (show->parsed()) runShow(showOptions);
  else if (splice->parsed()) runSplice(spliceOptions);
  else if (measurePatch->parsed()) runMeasurePatch(measurePatchOptions);
  else if (measurePair->parsed()) runMeasurePair(measurePairOptions);
  else if (samplingLibrary->parsed()) runSamplingLibrary(samplingOptions);
  return 0;
}
