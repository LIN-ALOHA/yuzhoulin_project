def get_classtable():
    classes = []
    with open("malware_family.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes

if __name__=='__main__':
    classes=get_classtable()
    print(classes)
