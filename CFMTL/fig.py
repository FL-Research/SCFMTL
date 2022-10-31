import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--experiment', type=str, default='performance-mnist')
parser.add_argument('--filename', type=str, default='fig-mnist-iid')

def figure(args):
    record = np.load('../experiments/{}.npy'.format(args.filename))
    record = record.tolist()

    if args.experiment == 'performance-mnist':
        x_0 = x_1 = x_2 = range(len(record[0][0]))
        y1_0 = record[0][0]
        y1_1 = record[0][1]
        y1_2 = record[0][2]
        y2_0 = record[1][0]
        y2_1 = record[1][1]
        y2_2 = record[1][2]

        plt.figure(figsize=(6.4, 12.8))

        plt.subplot(2, 1, 1)
        plt.plot(x_0, y1_0, '-', marker='o', mec='r', mfc='w', label='FedAvg', color='r', markersize=10)
        plt.plot(x_1, y1_1, '-', marker='s', mec='g', mfc='w', label='CFL', color='g', markersize=10)
        plt.plot(x_2, y1_2, '-', marker='^', mec='b', mfc='w', label='CFMTL', color='b', markersize=10)
        plt.ylabel('accuracy', fontsize=20)
        plt.ylim(80, 100)

        plt.subplot(2, 1, 2)
        plt.plot(x_0, y2_0, '-', marker='o', mec='r', mfc='w', label='FedAvg', color='r', markersize=10)
        plt.plot(x_1, y2_1, '-', marker='s', mec='g', mfc='w', label='CFL', color='g', markersize=10)
        plt.plot(x_2, y2_2, '-', marker='^', mec='b', mfc='w', label='CFMTL', color='b', markersize=10)
        plt.ylabel('proportion', fontsize=20)
        plt.ylim(0, 100)
        plt.xlabel('round', fontsize=20)

        plt.legend(fontsize=20)
        print(y1_0[-1])
        print(y1_1[-1])
        print(y1_2[-1])
        print(y2_0[-1])
        print(y2_1[-1])
        print(y2_2[-1])

    if args.experiment == 'performance-cifar':
        x_0 = x_1 = x_2 = range(len(record[0][0]))
        y1_0 = record[0][0]
        y1_1 = record[0][1]
        y1_2 = record[0][2]

        plt.figure(figsize=(6.4, 6.4))

        plt.subplot(1, 1, 1)
        plt.plot(x_0, y1_0, '-', marker='o', mec='r', mfc='w', label='FedAvg', color='r', markersize=10)
        plt.plot(x_1, y1_1, '-', marker='s', mec='g', mfc='w', label='CFL', color='g', markersize=10)
        plt.plot(x_2, y1_2, '-', marker='^', mec='b', mfc='w', label='CFMTL', color='b', markersize=10)
        plt.ylabel('accuracy', fontsize=20)
        plt.ylim(0, 100)

        plt.legend(fontsize=20)
        print(y1_0[-1])
        print(y1_1[-1])
        print(y1_2[-1])

    if args.experiment == 'communication':

        y_0 = record[1][1]
        y_1 = record[1][0]
        x_0 = record[0][1]
        x_1 = record[0][0]

        plt.figure(figsize=(6.4, 6.4))

        plt.subplot(1, 1, 1)
        plt.plot(x_0, y_0, '-', marker='o', mec='r', mfc='w', label='FMTL', color='r', markersize=10)
        plt.plot(x_1, y_1, '-', marker='s', mec='g', mfc='w', label='CFMTL', color='g', markersize=10)
        plt.xlabel('cost', fontsize=20)
        plt.xlim(0, 700)
        plt.ylabel('accuracy', fontsize=20)
        plt.ylim(0, 100)

        plt.legend(fontsize=20)
        for i in range(len(x1_0)):
            if x_0[i] >= 100:
                print(y_0[i])
                break
        for i in range(len(x1_1)):
            if x_1[i] >= 100:
                print(y_1[i])
                break

    if args.experiment == 'hyperparameters':
        x_0 = x_1 = x_2 = range(len(record[0][0]))
        y1_0 = record[0][0]
        y1_1 = record[0][1]
        y1_2 = record[0][2]
        y2_0 = record[1][0]
        y2_1 = record[1][1]
        y2_2 = record[1][2]

        plt.figure(figsize=(6.4, 12.8))

        plt.subplot(2, 1, 1)
        plt.plot(x_0, y1_0, '-', marker='o', mec='r', mfc='w', label='FedAvg', color='r', markersize=10)
        plt.plot(x_1, y1_1, '-', marker='s', mec='g', mfc='w', label='CFL', color='g', markersize=10)
        plt.plot(x_2, y1_2, '-', marker='^', mec='b', mfc='w', label='CFMTL', color='b', markersize=10)
        plt.ylabel('accuracy', fontsize=20)
        plt.ylim(80, 100)

        plt.subplot(2, 1, 2)
        plt.plot(x_0, y2_0, '-', marker='o', mec='r', mfc='w', label='FedAvg', color='r', markersize=10)
        plt.plot(x_1, y2_1, '-', marker='s', mec='g', mfc='w', label='CFL', color='g', markersize=10)
        plt.plot(x_2, y2_2, '-', marker='^', mec='b', mfc='w', label='CFMTL', color='b', markersize=10)
        plt.ylabel('proportion', fontsize=20)
        plt.ylim(0, 100)
        plt.xlabel('round', fontsize=20)

        plt.legend(fontsize=20)
        print(y1_0[-1])
        print(y1_1[-1])
        print(y1_2[-1])
        print(y2_0[-1])
        print(y2_1[-1])
        print(y2_2[-1])

    if args.experiment == 'metric':
        x_0 = x_1 = x_2 = x_3 = range(len(record[0][0]))
        y1_0 = record[0][0]
        y1_1 = record[0][1]
        y1_2 = record[0][2]
        y1_3 = record[0][3]
        y2_0 = record[1][0]
        y2_1 = record[1][1]
        y2_2 = record[1][2]
        y2_3 = record[1][3]

        plt.figure(figsize=(6.4, 12.8))

        plt.subplot(2, 1, 1)
        plt.plot(x_0, y1_0, '-', marker='o', mec='r', mfc='w', label='L2', color='r', markersize=10)
        plt.plot(x_1, y1_1, '-', marker='s', mec='g', mfc='w', label='Equal', color='g', markersize=10)
        plt.plot(x_2, y1_2, '-', marker='^', mec='b', mfc='w', label='L1', color='b', markersize=10)
        plt.plot(x_3, y1_3, '-', marker='h', mec='y', mfc='w', label='Cos', color='y', markersize=10)
        plt.ylabel('accuracy', fontsize=20)
        plt.ylim(80, 100)

        plt.subplot(2, 1, 2)
        plt.plot(x_0, y2_0, '-', marker='o', mec='r', mfc='w', label='L2', color='r', markersize=10)
        plt.plot(x_1, y2_1, '-', marker='s', mec='g', mfc='w', label='Equal', color='g', markersize=10)
        plt.plot(x_2, y2_2, '-', marker='^', mec='b', mfc='w', label='L1', color='b', markersize=10)
        plt.plot(x_3, y2_3, '-', marker='h', mec='y', mfc='w', label='Cos', color='y', markersize=10)
        plt.ylabel('proportion', fontsize=20)
        plt.ylim(0, 100)
        plt.xlabel('round', fontsize=20)

        plt.legend(fontsize=20)
        print(y1_0[-1])
        print(y1_1[-1])
        print(y1_2[-1])
        print(y1_3[-1])
        print(y2_0[-1])
        print(y2_1[-1])
        print(y2_2[-1])
        print(y2_3[-1])

    plt.savefig('../experiments/{}.eps'.format(args.filename))
    plt.savefig('../experiments/{}.pdf'.format(args.filename))


def figure2(data_path, save_path, choose:int):


    path1, path2, path3, path4 = data_path[0], data_path[1], data_path[2], data_path[3]
    if choose == 0:
        record1 = np.load(path1, allow_pickle=True)
        record2 = np.load(path2, allow_pickle=True)
        record3 = np.load(path3, allow_pickle=True)[0][2]
        record4 = np.load(path4, allow_pickle=True)[0][2]
    else:
        record1 = np.load(path1, allow_pickle=True)[0][0]
        record2 = np.load(path2, allow_pickle=True)[0][1]
        record3 = np.load(path3, allow_pickle=True)[0][2]
        record4 = np.load(path4, allow_pickle=True)[0][2]

    markersize = 6
    markevery = 8
    linestyle = '-.'
    color_set = {"red": "#f4433c", "green": "#0aa858", "blue": "#2d85f0", "yellow": "#ffbc32"}
    
    plt.figure(figsize=(6.4, 6.4))
    plt.grid(linestyle=linestyle)

    plt.plot(record1, '-', marker='o', markevery=markevery, mec='r', mfc='w', color=color_set["red"], markersize=markersize, label="FedAvg")
    plt.plot(record2, '-', marker='s', markevery=markevery, mec='y', mfc='w', color=color_set["yellow"], markersize=markersize, label="CFL")
    plt.plot(record3, '-', marker='^', markevery=markevery, mec='b', mfc='w', color=color_set["blue"], markersize=markersize, label="CFMTL")
    plt.plot(record4, '-', marker='>', markevery=markevery, mec='g', mfc='w', color=color_set["green"], markersize=markersize, label="SCFMTL")

    plt.xlabel("round", fontsize=20)
    plt.ylabel('accuracy', fontsize=20)

    plt.ylim(70, 100)

    plt.legend(fontsize=20)
    plt.savefig(f"result/pdf/{save_path}.pdf")

if __name__ == '__main__':
    # args = parser.parse_args()



    # data_path_list = [
    #    ["result/fedavg_iid.npy", "result/CFL_iid.npy", "result/caltech101_CFMTL_iid.npy", "result/caltech101_CFMTL_iid_L_0.1.npy"],
    #    ["result/fedavg_0.25.npy", "result/CFL_0.25.npy", "result/caltech101_CFMTL_0.25.npy", "result/caltech101_CFMTL_0.25_L0.1.npy"] ,
    #    ["result/fedavg_0.5.npy", "result/caltech101_0.5_1.npy", "result/caltech101_CFMTL_0.50.npy", "result/caltech101_CFMTL_0.50_L0.1.npy"],
    # ]

    data_path_list = [
        [
            "result/caltech101_0.75.npy", 
            "log_simple_net/result/caltech101_CFMTL_0.75_equal_L1_sgd.npy", 
            "log_simple_net/result/caltech101_CFMTL_0.75_equal_L1_sgd.npy",
            "log_simple_net/result/caltech101_CFMTL_0.75_equal_L0.1_sgd.npy"
        ],
        [
            "result/caltech101_sc.npy", 
            "result/caltech101_CFMTL_sc.npy", 
            "result/caltech101_CFMTL_sc.npy", 
            "result/caltech101_CFMTL_sc_L0.1.npy"
        ]
    ]
    # save_path_list = ["caltech_iid", "caltech_non_iid_0.25", "caltech_non_iid_0.50"]
    save_path_list = ["caltech_non_iid_0.75_1", "caltech_non_iid_single_class_1"]

    choose = 1

    for i in range(len(data_path_list)):
        figure2(data_path_list[i], save_path_list[i], choose)
    
