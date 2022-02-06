import os, sys

tests = []

for f in filter(lambda fn: "test_" in fn, os.listdir(".")):
    with open(f) as log:
        content = log.read()
    if "FINAL" not in content:
        continue

    text = content[content.find("FINAL"):]
    text = text[:text.find("\n")]

    cfg = []
    search = open("search.py").read()
    for x in content[:content.find("Policy")].split("\n"):
        var = x[:x.find("=")].strip()
        if '"%s"'%var not in search or not len(var):
            continue
        cfg.append(x.strip())

    score = float(text[text.find(" ")+1:].strip())
    watch = "[{}] -> <{}>".format(text, f)
    info = " CFG\n{}\n{}".format(';'.join(cfg), '*' * 50)

    tests.append([score, info, watch])

tests.sort()
for (_, i, w) in reversed(tests[-6:]):
    if len(sys.argv) > 1:
        print(w, i)
    else:
        print(w)
