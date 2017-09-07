# characterize.py

import argparse
import codecs


def process(n1, n2, tok=None):
    fd = codecs.open(n1, "r", "utf-8")
    fw = codecs.open(n2, "w", "utf-8")

    for line in fd:
        line = line.strip()
        wlist = line.split()
        clist = []

        for word in wlist:
            for char in word:
                clist.append(char)
            if tok != None:
                clist.append(tok)

        fw.write(" ".join(clist) + "\n")

    fd.close()
    fw.close()


def parseargs():
    msg = "characterize corpus"
    parser = argparse.ArgumentParser(description=msg)

    msg = "corpus"
    parser.add_argument("--corpus", required=True, help=msg)
    msg = "output"
    parser.add_argument("--output", required=True, help=msg)
    msg = "add word seperation token"
    parser.add_argument("--token", type=str, help=msg)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    process(args.corpus, args.output, args.token)
