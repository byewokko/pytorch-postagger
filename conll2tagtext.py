import sys

wordlimit = 50

sent = []
for line in sys.stdin:
    if line.startswith("#"):
        continue
    if not line.strip():
        if len(sent) <= wordlimit:
            print(" ".join(sent))
        sent = []
        continue
    line = line.strip().split("\t")
    if "-" in line[0]:
        continue
    sent.append("_".join((line[1].replace("_","-"), line[3])))

print(" ".join(sent))