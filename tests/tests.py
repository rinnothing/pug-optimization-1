import optimize.golden_search
import optimize.random_search
import common.tests_function


def get_eps_stop_determiner(eps: float):
    """
    eps bounds difference stopper, only works for binary search
    :param eps: value, that determines what bounds difference it should stop
    :return: to stop or not
    """

    def determiner(state: common.OptimizeResult) -> bool:
        return (state.history[-1][1] - state.history[-1][0]) < eps

    return determiner


def print_check_random_search(function, lim, count):
    print("Random search result(" , count, " launches):")
    first_res = optimize.random_search.random_search(function, stop=get_eps_stop_determiner(0.01),
                                                     bounds=lim)
    first_find_res = function(first_res.get_res())
    print("First start:")
    print("Count of function calls: ", first_res.count_of_function_calls, " | result: ", first_find_res)
    res_with_min = [first_find_res, first_res.count_of_function_calls]
    res_with_max = [first_find_res, first_res.count_of_function_calls]
    res_with_min_call = [first_find_res, first_res.count_of_function_calls]
    res_with_max_call = [first_find_res, first_res.count_of_function_calls]
    sum_res = [first_find_res, first_res.count_of_function_calls]
    for i in range(1, count):
        new_res = optimize.random_search.random_search(function, stop=get_eps_stop_determiner(0.01),
                                                     bounds=lim)
        new_res_pair=[function(new_res.get_res()), new_res.count_of_function_calls]
        sum_res = [sum_res[0] + new_res_pair[0], sum_res[1] + new_res_pair[1]]
        if res_with_min[0] > new_res_pair[0]:
            res_with_min = new_res_pair
        if res_with_max[0] < new_res_pair[0]:
            res_with_max = new_res_pair
        if res_with_min_call[1] > new_res_pair[1]:
            res_with_min_call = new_res_pair
        if res_with_max_call[1] < new_res_pair[1]:
            res_with_max_call = new_res_pair
    print("Minimum result:")
    print("Count of function calls: ", res_with_min[1], " | result: ", res_with_min[0])
    print("Maximum result:")
    print("Count of function calls: ", res_with_max[1], " | result: ", res_with_max[0])
    print("Minimum function calls result:")
    print("Count of function calls: ", res_with_min_call[1], " | result: ", res_with_min_call[0])
    print("Maximum function calls result:")
    print("Count of function calls: ", res_with_min_call[1], " | result: ", res_with_min_call[0])
    print("Middle result:")
    print("Count of function calls: ", sum_res[1]/count, " | result: ", sum_res[0]/count)


def check_functions(functions):
    i = 0
    for test_function in functions:
        i += 1
        lim = test_function.lim
        print("-------------------------------------")
        print("Function number ", i, ":")
        print("Golden search result: ")
        result = optimize.golden_search.golden_search(test_function.function, stop=get_eps_stop_determiner(0.01),
                                                      bounds=lim)
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", test_function.function(result.get_res()))
        print_check_random_search(test_function.function, lim, 100)

print("Functions with one min:")
check_functions(common.tests_function.functions_with_one_min)
print("")
print("")
print("Functions with local min:")
check_functions(common.tests_function.functions_with_local_min)