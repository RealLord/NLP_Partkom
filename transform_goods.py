import csv


def transform_row(line, fo2):
    if line[5] != '0':
        #if line[1].find('еталь') == -1:
        if line[4] != '0':
            nstr = line[1].replace('""', '"')
#                nstr = nstr.replace("\"", "")
            fo2.writelines('"'+line[4]+'";'+'"'+line[3]+'";'+'"'+nstr+'";'+'"'+line[2]+'";'+'"'+line[5]+'";'+"\n")
            return 1
    return 0


def second_operation():
    fo2 = open('goods_01.csv', 'w', encoding='utf-8')
    fo2.writelines('"artical";"brend_code";"desc";"guid";"group_code";'+"\n")

    with open("goods_00.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_ALL)
        index = 1
        skipped = 0
        written = 0
        for row in reader:
            if reader.line_num > 1:
                code = transform_row(row, fo2)
                if code == 0:
                    skipped = skipped + 1
                else:
                    written = written + 1
                if index % 100000 == 0:
                    print("all = %s, skipped = %s, written = %s" % (index, skipped, written))
                index = index + 1
    fo2.close()


def first_operation():
    fi1 = open("goods.csv", "r", encoding='utf-16')
    fo1 = open('goods_00.csv', 'w', encoding='utf-8')
    index = 1
    while True:
        line = fi1.readline()
        if not line:
            break
        new_str0 = line.replace('""""', '"', 50)
        new_str1 = new_str0.replace('"""', '"', 50)
        new_str2 = new_str1.replace(';"";', ';"  ";', 50)
        new_str2 = new_str2.replace(';"";', ';"  ";', 50)
        new_str2 = new_str2.replace(';"";', ';"  ";', 50)
        new_str2 = new_str2.replace("\"\"\n", "\" \"\n", 50)
        new_str3 = new_str2.replace('""', '"', 50)
        if new_str3.find('еталь') == -1:
            fo1.writelines(new_str3)
        if index % 100000 == 0:
            print("processed = %s" % index)
        index = index + 1

    print("processed = %s" % index)
    fi1.close()
    fo1.close()

def check_file():
    fi1 = open("goods_00.csv", "r", encoding='utf-8')
    index = 1
    detal = 0
    while True:
        line = fi1.readline()
        if not line:
            break
        if line.find('еталь') != -1:
            detal = detal + 1
        index = index + 1

    print("всего = %s, детали = %s" % (index, detal))
    fi1.close()

if __name__ == '__main__':
    #check_file()
    #first_operation()
    second_operation()



