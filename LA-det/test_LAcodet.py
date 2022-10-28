import argparse
import os
from copy import deepcopy

import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader

from coperception.datasets import LA_V2XSimDet
from coperception.utils.LACoDetModule import *
from coperception.configs import Config, ConfigGlobal
# from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.utils.mean_ap import eval_map
from coperception.models.det import *
from coperception.models.LAdet import *
from coperception.utils.detection_util import late_fusion
from coperception.utils.data_util import apply_pose_noise


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


@torch.no_grad()
def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)
    center_agent=1
    config.LA_Flag = args.LA_Flag
    config.k = args.k
    config.compensation_flag = args.compensation_flag
    config.tau = args.tau
    need_log = args.log
    num_workers = args.nworker
    apply_late_fusion = args.apply_late_fusion
    pose_noise = args.pose_noise
    compress_level = args.compress_level
    only_v2i = args.only_v2i

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    config.inference = args.inference
    if args.com == "upperbound":
        flag = "upperbound"
    elif args.com == "when2com":
        flag = "when2com"
        if args.inference == "argmax_test":
            flag = "who2com"
        if args.warp_flag:
            flag = flag + "_warp"
    elif args.com in {"v2v", "disco", "syncdisco", "sum", "mean", "max", "cat", "agent"}:
        flag = args.com
    elif args.com == "lowerbound":
        flag = "lowerbound"
        if args.box_com:
            flag += "_box_com"
    else:
        raise ValueError(f"com: {args.com} is not supported")

    print("flag", flag)
    config.flag = flag
    config.split = "test"

    num_agent = args.num_agent
    # agent0 is the RSU
    agent_idx_range = range(num_agent) if args.rsu else range(1, num_agent)
    # agent_idx_range = 1
    validation_dataset = LA_V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound="upperbound" if args.com == "upperbound" else "lowerbound",
        kd_flag=args.kd_flag,
        k=args.k,
        tau=config.tau,
        rsu=args.rsu,
    )
    validation_data_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    print("Validation dataset size:", len(validation_dataset))

    if not args.rsu:
        num_agent -= 1

    if flag == "upperbound" or flag.startswith("lowerbound"):
        model = FaFNet(
            config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent
        )
    elif flag.startswith("when2com") or flag.startswith("who2com"):
        # model = PixelwiseWeightedFusionSoftmax(config, layer=args.layer)
        model = When2com(
            config,
            layer=args.layer,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "disco":
        model = DiscoNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "syncdisco":
        model = SyncDiscoNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "sum":
        model = SumFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "mean":
        model = MeanFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "max":
        model = MaxFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "cat":
        model = CatFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "agent":
        model = AgentWiseWeightedFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "v2v":
        model = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)
    if args.TM_flag == 1:
        fafmodule.model.module.compensation.TM_Flag=True
    model_save_path = args.resume[: args.resume.rfind("/")]

    if args.inference == "argmax_test":
        model_save_path = model_save_path.replace("when2com", "who2com")

    os.makedirs(model_save_path, exist_ok=True)
    log_file_name = os.path.join(model_save_path, "log.txt")
    saver = open(log_file_name, "a")
    saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
    saver.flush()

    # Logging the details for this experiment
    saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
    saver.write(args.__repr__() + "\n\n")
    saver.flush()

    checkpoint = torch.load(
        args.resume, map_location="cpu"
    )  # We have low GPU utilization for testing
    start_epoch = checkpoint["epoch"] + 1
    fafmodule.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    # fafmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if start_epoch > 136:
        fafmodule.model.module.compensation.TM_Flag=True
    print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    #  ===== eval =====
    fafmodule.model.eval()
    save_fig_path = [
        check_folder(os.path.join(model_save_path, f"vis{i}")) for i in agent_idx_range
    ]
    tracking_path = [
        check_folder(os.path.join(model_save_path, f"tracking{i}"))
        for i in agent_idx_range
    ]

    # for local and global mAP evaluation
    # det_results_local = [[] for i in agent_idx_range]
    # annotations_local = [[] for i in agent_idx_range]
    det_results_local = [[]]
    annotations_local = [[]]

    tracking_file = [set()] * num_agent
    for cnt, sample in enumerate(validation_data_loader):
        t = time.time()
        (
            padded_voxel_point_list,
            padded_voxel_points_teacher_list,
            label_one_hot_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_map_list,
            vis_maps_list,
            gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list_supervision, # trans_matrices to get Real-Time H
            trans_matrices_list,  # matrix for coordinate transformation. e.g. [batch_idx, j, i] ==> transformation matrix totransfer from agent i to j
            trans_matrices_newest_list, # trans to the newest frame of each agent
            padded_voxel_point_supervision_list #voxelized point cloud for individual agent to supervision the results of 
        ) = zip(*sample)

        print(filenames)

        filename0 = filenames[0]
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        trans_matrices_supervision = torch.stack(tuple(trans_matrices_list_supervision), 1)
        trans_matrices_newest = torch.stack(trans_matrices_newest_list, 1)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_all_agents = torch.stack(tuple(num_agent_list), 1)

        # add pose noise
        if pose_noise > 0:
            apply_pose_noise(pose_noise, trans_matrices)

        if not args.rsu:
            num_all_agents -= 1

        if flag == "upperbound":
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        else:
            padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)
            padded_voxel_point_supervision = torch.cat(tuple(padded_voxel_point_supervision_list), 0)

        label_one_hot = label_one_hot_list[center_agent]
        reg_target = reg_target_list[center_agent]
        reg_loss_mask = reg_loss_mask_list[center_agent]
        anchors_map = anchors_map_list[center_agent]
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        data = {
            "bev_seq": padded_voxel_points.to(device),
            "labels": label_one_hot.to(device),
            "reg_targets": reg_target.to(device),
            "anchors": anchors_map.to(device),
            "vis_maps": vis_maps.to(device),
            "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
            "target_agent_ids": target_agent_ids.to(device),
            "num_agent": num_all_agents.to(device),
            "trans_matrices_supervision": trans_matrices_supervision,
            "trans_matrices": trans_matrices,
            "trans_matrices_newest": trans_matrices_newest,
            "bev_supervision": padded_voxel_point_supervision
        }

        if flag == "lowerbound_box_com":
            loss, cls_loss, loc_loss, result = fafmodule.predict_all_with_box_com(
                data, data["trans_matrices"]
            )
        elif flag == "disco":
            (
                loss,
                cls_loss,
                loc_loss,
                result,
                save_agent_weight_list,
            ) = fafmodule.predict_all(data, 1, num_agent=num_agent)
        else:
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                data, 1, num_agent=num_agent
            )

        box_color_map = ["red", "yellow", "blue", "purple", "black", "orange"]

        # If has RSU, do not count RSU's output into evaluation
        eval_start_idx = 1 if args.rsu else 0

        # local qualitative evaluation
        for k in range(1):
            box_colors = None
            if apply_late_fusion == 1 and len(result[k]) != 0:
                pred_restore = result[k][0][0][0]["pred"]
                score_restore = result[k][0][0][0]["score"]
                selected_idx_restore = result[k][0][0][0]["selected_idx"]

            data_agents = {
                "bev_seq": torch.unsqueeze(padded_voxel_points[k+1, :, :, :, :], 1),
                "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
                "anchors": torch.unsqueeze(anchors_map[k, :, :, :, :], 0),
            }
            temp = gt_max_iou[k+1]

            if len(temp[0]["gt_box"]) == 0:
                data_agents["gt_max_iou"] = []
            else:
                data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]

            # late fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                box_colors = late_fusion(
                    k, num_agent, result, trans_matrices, box_color_map
                )

            result_temp = result[k]

            temp = {
                "bev_seq": data_agents["bev_seq"][-1, -1].cpu().numpy(),
                "result": [] if len(result_temp) == 0 else result_temp[0][0],
                "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
                "anchors_map": data_agents["anchors"].cpu().numpy()[0],
                "gt_max_iou": data_agents["gt_max_iou"],
            }
            det_results_local[k], annotations_local[k] = cal_local_mAP(
                config, temp, det_results_local[k], annotations_local[k]
            )

            filename = str(filename0[0][0])
            cut = filename[filename.rfind("agent") + 7 :]
            seq_name = cut[: cut.rfind("_")]
            idx = cut[cut.rfind("_") + 1 : cut.rfind("/")]
            seq_save = os.path.join(save_fig_path[k], seq_name)
            check_folder(seq_save)
            idx_save = str(idx) + ".png"
            temp_ = deepcopy(temp)
            if args.visualization:
                print(os.path.join(seq_save, idx_save))
                visualization(
                    config,
                    temp,
                    box_colors,
                    box_color_map,
                    apply_late_fusion,
                    os.path.join(seq_save, idx_save),
                )

            # # plot the cell-wise edge
            # if flag == "disco" and k < len(save_agent_weight_list):
            #     one_agent_edge = save_agent_weight_list[k]
            #     for kk in range(len(one_agent_edge)):
            #         idx_edge_save = (
            #             str(idx) + "_edge_" + str(kk) + "_to_" + str(k) + ".png"
            #         )
            #         savename_edge = os.path.join(seq_save, idx_edge_save)
            #         sns.set()
            #         plt.savefig(savename_edge, dpi=500)
            #         plt.close(0)

            # == tracking ==
            if args.tracking:
                scene, frame = filename.split("/")[-2].split("_")
                det_file = os.path.join(tracking_path[k], f"det_{scene}.txt")
                if scene not in tracking_file[k]:
                    det_file = open(det_file, "w")
                    tracking_file[k].add(scene)
                else:
                    det_file = open(det_file, "a")
                det_corners = get_det_corners(config, temp_)
                for ic, c in enumerate(det_corners):
                    det_file.write(
                        ",".join(
                            [
                                str(
                                    int(frame) + 1
                                ),  # frame idx is 1-based for tracking
                                "-1",
                                "{:.2f}".format(c[0]),
                                "{:.2f}".format(c[1]),
                                "{:.2f}".format(c[2]),
                                "{:.2f}".format(c[3]),
                                str(result_temp[0][0][0]["score"][ic]),
                                "-1",
                                "-1",
                                "-1",
                            ]
                        )
                        + "\n"
                    )
                    det_file.flush()

                det_file.close()

            # restore data before late-fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                result[k][0][0][0]["pred"] = pred_restore
                result[k][0][0][0]["score"] = score_restore
                result[k][0][0][0]["selected_idx"] = selected_idx_restore

        print("Validation scene {}, at frame {}".format(seq_name, idx))
        print("Takes {} s\n".format(str(time.time() - t)))

    logger_root = args.logpath if args.logpath != "" else "logs"
    logger_root = os.path.join(
        logger_root, f"{flag}_eval", "with_rsu" if args.rsu else "no_rsu"
    )
    os.makedirs(logger_root, exist_ok=True)
    log_file_path = os.path.join(logger_root, "log_test.txt")
    log_file = open(log_file_path, "a")

    def print_and_write_log(log_str):
        print(log_str)
        log_file.write(log_str + "\n")

    mean_ap_local = []
    # local mAP evaluation
    det_results_all_local = []
    annotations_all_local = []
    # for k in range(eval_start_idx, num_agent):
    for k in range(1):
        if type(det_results_local[k]) != list or len(det_results_local[k]) == 0:
            continue

        print_and_write_log("Local mAP@0.5 from agent {}".format(k))
        mean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap)
        print_and_write_log("Local mAP@0.7 from agent {}".format(k))

        ean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.7,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap)

        det_results_all_local += det_results_local[k]
        annotations_all_local += annotations_local[k]

    mean_ap_local_average, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.5,
        dataset=None,
        logger=None,
    )
    mean_ap_local.append(mean_ap_local_average)

    mean_ap_local_average, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.7,
        dataset=None,
        logger=None,
    )
    mean_ap_local.append(mean_ap_local_average)

    print_and_write_log(
        "Quantitative evaluation results of model from {}, at epoch {} at latency contition {}".format(
            args.resume, start_epoch - 1, config.tau
        )
    )

    # for k in range(eval_start_idx, num_agent):
    for k in range(1):
        print_and_write_log(
            "agent{} mAP@0.5 is {} and mAP@0.7 is {}".format(
                k + 1 if not args.rsu else k, mean_ap_local[k * 2], mean_ap_local[(k * 2) + 1]
            )
        )

    print_and_write_log(
        "average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(
            mean_ap_local[-2], mean_ap_local[-1]
        )
    )

    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default='/GPFS/public/V2XSim2LA/val',
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=1, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")    
    parser.add_argument('--tau', default=[0,0,0,0,0,0], nargs='+', type=int, help='Latency for each agent')
    parser.add_argument("--k", default=3, type=int, help="How many frames do you want to use in compensation")
    parser.add_argument("--LA_Flag", default=True, type=bool, help="whether enable LA")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument(
        "--compensation_flag",
        default="Baseline",
        type=str,
        help="which compensation method to use",
    )
    parser.add_argument(
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--warp_flag", default=0, type=int, help="Whether to use pose info for When2com"
    )
    parser.add_argument(
        "--TM_flag", default=0, type=int, help="Whether to use Time Modulation."
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--visualization", type=int, default=0, help="Visualize validation result"
    )
    parser.add_argument(
        "--com",
        default="syncdisco",
        type=str,
        help="lowerbound/upperbound/disco/when2com/v2v/sum/mean/max/cat/agent",
    )
    parser.add_argument("--inference", type=str)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--box_com", action="store_true")
    parser.add_argument("--rsu", default=1, type=int, help="0: no RSU, 1: RSU")
    # scene_batch => batch size in each scene
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--apply_late_fusion",
        default=0,
        type=int,
        help="1: apply late fusion. 0: no late fusion",
    )
    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)
