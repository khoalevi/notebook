import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True, help="name of the user")
args = vars(parser.parse_args())

print("hi there {}, it's nice to meet you!".format(args["name"]))