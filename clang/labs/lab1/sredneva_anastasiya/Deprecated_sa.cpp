#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class DeprecatedVisitor : public RecursiveASTVisitor<DeprecatedVisitor> {
public:
  explicit DeprecatedVisitor(ASTContext *Context) : Context(Context) {}
  bool VisitFunctionDecl(FunctionDecl *fDecl) {
    if (fDecl->getNameInfo().getAsString().find("deprecated") !=
        std::string::npos) {
      DiagnosticsEngine &diagn = Context->getDiagnostics();
      unsigned diagnID = diagn.getCustomDiagID(
          DiagnosticsEngine::Warning, "The function name has 'deprecated'");
      diagn.Report(fDecl->getLocation(), diagnID)
          << fDecl->getNameInfo().getAsString();
    }
    return true;
  }

private:
  ASTContext *Context;
};

class DeprecatedConsumer : public ASTConsumer {
public:
  explicit DeprecatedConsumer(ASTContext *Context) : Visitor(Context) {}
  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  DeprecatedVisitor Visitor;
};

class DeprecatedAction : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<DeprecatedConsumer>(&Compiler.getASTContext());
  }
  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }
};

static FrontendPluginRegistry::Add<DeprecatedAction>
    X("deprecated_plugin",
      "adds warning if there is a 'deprecated' in the function name");
