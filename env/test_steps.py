import csv
with open('/home/zzc/PycharmProjects/pybullet_neoschunk_reaching/results/steps_20211208231135_prio.csv', newline='') as f:
    reader = csv.reader(f)
    sum = 0
    isum = 0
    for row in reader:
        #print(row[1])
        sum += int(row[1])
        isum += 1
        if isum==900:
            break
    print(sum, isum, sum/isum)


    # /home/zzc/PycharmProjects/pybullet_neoschunk_reaching/logs/1201wall03p, prio : epi 932, step 455; epi 140, step 537
    # /home/zzc/PycharmProjects/pybullet_neoschunk_reaching/logs/1201wall03, free : epi 1794, step 519; epi 153, step 549