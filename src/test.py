import fire

def main(name, arr, val):
    print("Name: {}".format(name))
    print("Val: {}".format(val))
    print("ARR: {} of {}".format(type(arr), arr))

if __name__ == "__main__":
    fire.Fire(main)