singleton(x) = [x]
@dist exactly(x) = singleton(x)[uniform_discrete(1, 1)]
@dist uniform_from_list(list) = list[uniform_discrete(1, length(list))]
@dist categorical_from_list(list, probs) = list[categorical(probs)]