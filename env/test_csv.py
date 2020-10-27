import csv
with open('/home/zheng/resultslog/results/steps_202010091003_free.csv', newline='') as f:
    reader = csv.reader(f)
    sum = 0
    isum = 0
    for row in reader:
        #print(row[1])
        sum += int(row[1])
        isum += 1
    print(sum, isum, sum/isum)