import numpy as np
import common.tests_function
import scipy.optimize as opt
import matplotlib
matplotlib.use("TkAgg")

def golden_methods(func):
    i = 0
    for test_func in func:
        i += 1
        lim = test_func.lim

        golden_result = opt.minimize_scalar(test_func.function, bounds=lim, method='bounded')
        print(f"Function number {i}, Bounds: {lim}")
        print(f"Golden Search: Calls: {golden_result.nfev}, Result: {golden_result.x}")

        if not golden_result.success:
            print("Didn't solve")
def grad_methods(func):
    for index, test_func in enumerate(func):
        lim = test_func.lim
        f = test_func.function
        gr = test_func.gradient
        x0 = np.random.uniform(-5, 10)

        grad_result = opt.minimize(f, x0=x0, method='CG', jac=gr)

        print(f"Function number", index + 1, "Bounds:", lim)
        print("Conjugate Gradient Descent: Calls:", grad_result.nfev, "Result:", grad_result.x)

        if not grad_result.success:
            print("Didn't solve")
if __name__ == "__main__":
    print("Running Golden Section Search with one min:")
    golden_methods(common.tests_function.functions_with_one_min)
    print("-" * 50)
    print("Running Golden Section Search with local min:")
    golden_methods(common.tests_function.functions_with_local_min)
    print("-" * 50)
    print("\nRunning Gradient Descent:")
    grad_methods(common.tests_function.functions_with_one_min)
    print("-" * 50)
    print("\nRunning Gradient Descent:")
    grad_methods(common.tests_function.functions_with_local_min)

