"""Hello world example"""

import tweezer

print("Welcome to tweezer version {}!".format(tweezer.__version__))

tweezer.set_verbose(1)
tweezer.function1((1,2))