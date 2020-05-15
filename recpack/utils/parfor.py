from joblib import Parallel, delayed


def to_tuple(el):
    if type(el) == tuple:
        return el
    else:
        return (el, )


def parfor(iterator, parallel=True, n_jobs=-1, **kwargs):
    """
    Decorator to turn a for loop into a parallel call from joblib.
    The following code fragments are identical:

    l = list()
    for i in range(5):
        l.append(i)
    # l = [0, 1, 2, 3, 4]

    @parfor(range(5))
    def l(i):
        return i
    # l = [0, 1, 2, 3, 4]

    Do note that, depending on the backend, changes to variables outside the loop will not be reflected!

    Example usage:
    @parfor(enumerate(range(5, 50)))
    def f(i, c):
        time.sleep(1)
        print(i, c)
        return


    @parfor(range(10), parallel=False)
    def f(i):
        time.sleep(1)
        print(i)
    """
    def decorator(body):

        if parallel:
            data = Parallel(n_jobs=n_jobs, **kwargs)(delayed(body)(*to_tuple(args)) for args in iterator)
            return data
        else:
            # could replace manually doing this with backend=sequential
            data = list()
            for args in iterator:
                data.append(body(*to_tuple(args)))

        return data

    return decorator
