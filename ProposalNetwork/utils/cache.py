import os
import pickle

def cache(func):
    '''caches the output of the function and saves it to a pickle file with name 'cache/{function_name}.pkl',
      the next time it is called the result is loaded from the pickle file'''
    os.makedirs('cache', exist_ok=True)
    file = f'cache/{func.__name__}.pkl'
    def wrapper(*args, **kwargs):
        if not os.path.exists(file):
            var = func(*args, **kwargs)
            # do thing
            with open(file, "wb") as f:
                pickle.dump(var, f)
        else:
            with open(file, "rb") as f:
                var = pickle.load(f)
        return var
    return wrapper

# example use
if __name__ == '__main__':

    @cache
    def hello(num):
        return num + 2

    print(hello(4))