// RUN: %clang_cc1 -load %llvmshlibdir/Deprecated_sa%pluginext -plugin deprecated_plugin %s 2>&1 | FileCheck %s

// CHECK: warning: The function name has 'deprecated'
void deprecated();

// CHECK: warning: The function name has 'deprecated'
void deprecatedFunc();

// CHECK: warning: The function name has 'deprecated'
int deprecatedSumm(int a, int b) {
	return a + b;
}

// CHECK-NOT: warning: The function name has 'deprecated'
void deprecation();

// CHECK-NOT: warning: The function name has 'deprecated'
void deprfunction();

// CHECK-NOT: warning: The function name has 'deprecated'
void foo();