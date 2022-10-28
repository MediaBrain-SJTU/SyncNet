import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coperception.datasets import LA_V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.LACoDetModule import *
from coperception.utils.loss import *
from coperception.models.det import *
from coperception.models.LAdet import *
from coperception.utils import AverageMeter
from coperception.utils.data_util import apply_pose_noise

import glob
import os


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)
    config.LA_Flag = args.LA_Flag
    config.k = args.k
    config.compensation_flag = args.compensation_flag
    config.tau = args.tau
    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker
    start_epoch = 1
    batch_size = args.batch_size
    compress_level = args.compress_level
    auto_resume_path = args.auto_resume_path
    pose_noise = args.pose_noise
    only_v2i = args.only_v2i
    center_agent = args.center_agent

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.com == "upperbound":
        flag = "upperbound"
    elif args.com == "when2com" and args.warp_flag:
        flag = "when2com_warp"
    elif args.com in [
        "lowerbound",
        "v2v",
        "disco",
        "sum",
        "mean",
        "max",
        "cat",
        "agent",
        "when2com",
        "syncdisco"
    ]:
        flag = args.com
    else:
        raise ValueError(f"com: {args.com} is not supported")

    config.flag = flag

    num_agent = args.num_agent
    # agent0 is the RSU
    agent_idx_range = range(num_agent) if args.rsu else range(1, num_agent)
    training_dataset = LA_V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="train",
        bound="upperbound" if args.com == "upperbound" else "lowerbound",
        kd_flag=args.kd_flag,
        rsu=args.rsu,
        k=args.k,
        tau=config.tau
    )
    training_data_loader = DataLoader(
        training_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    print("Training dataset size:", len(training_dataset))

    logger_root = args.logpath if args.logpath != "" else "logs"

    if not args.rsu:
        num_agent -= 1

    if flag == "lowerbound" or flag == "upperbound":
        model = FaFNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
        )
    elif flag == "when2com" or flag == "when2com_warp":
        model = When2com(
            config,
            layer=args.layer,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "v2v":
        model = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "disco":
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
    elif flag == "sum":
        model = SumFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "mean":
        model = MeanFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "max":
        model = MaxFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "cat":
        model = CatFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif flag == "agent":
        model = AgentWiseWeightedFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    if args.kd_flag == 1:
        teacher = TeacherNet(config)
        teacher = nn.DataParallel(teacher)
        teacher = teacher.to(device)
        faf_module = FaFModule(
            model, teacher, config, optimizer, criterion, args.kd_flag
        )
        checkpoint_teacher = torch.load(args.resume_teacher)
        start_epoch_teacher = checkpoint_teacher["epoch"]
        faf_module.teacher.load_state_dict(checkpoint_teacher["model_state_dict"])
        print(
            "Load teacher model from {}, at epoch {}".format(
                args.resume_teacher, start_epoch_teacher
            )
        )
        faf_module.teacher.eval()
    else:
        faf_module = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    rsu_path = "with_rsu" if args.rsu else "no_rsu"
    model_save_path = check_folder(logger_root)
    model_save_path = check_folder(os.path.join(model_save_path, flag))

    if args.rsu:
        model_save_path = check_folder(os.path.join(model_save_path, "with_rsu"))
    else:
        model_save_path = check_folder(os.path.join(model_save_path, "no_rsu"))

    # check if there is valid check point file
    has_valid_pth = False
    if args.resume != ' ':
        for pth_file in os.listdir(os.path.join(auto_resume_path, f"{flag}/{rsu_path}")):
            if pth_file.startswith("epoch_") and pth_file.endswith(".pth"):
                has_valid_pth = True
                break
        auto_resume_path = ""

    if not has_valid_pth:
        print(
            f"No valid check point file in {auto_resume_path} dir, weights not loaded."
        )
        auto_resume_path = ""

    if args.resume == "" and auto_resume_path == "":
        log_file_name = os.path.join(model_save_path, "log.txt")
        saver = open(log_file_name, "w")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()
    else:
        if auto_resume_path != "":
            model_save_path = os.path.join(auto_resume_path, f"{flag}/{rsu_path}")
        else:
            model_save_path = args.resume[: args.resume.rfind("/")]

        print(f"model save path: {model_save_path}")

        log_file_name = os.path.join(model_save_path, "log.txt")

        if os.path.exists(log_file_name):
            saver = open(log_file_name, "a")
        else:
            os.makedirs(model_save_path, exist_ok=True)
            saver = open(log_file_name, "w")

        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        if auto_resume_path != "":
            list_of_files = glob.glob(f"{model_save_path}/*.pth")
            latest_pth = max(list_of_files, key=os.path.getctime)
            checkpoint = torch.load(latest_pth)
        else:
            checkpoint = torch.load(args.resume)

        # untrainable_list = ['decoder', 'encoder', 'classification', 'regression', 'fusion']
        untrainable_list = args.untrainable_list
        start_epoch = checkpoint["epoch"] + 1
        faf_module.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # faf_module.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        faf_module.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        for k,v in faf_module.model.named_parameters(): 
            # print(k[])
            for para in untrainable_list:
                if para in k: #or ('decoder' in k): 
                    v.requires_grad = False
        for k,v in faf_module.model.named_parameters(): 
            if 'transformer' in k: #or ('decoder' in k): 
                v.requires_grad = True
        faf_module.optimizer = optim.Adam(filter(lambda p: p.requires_grad, faf_module.model.parameters()), lr=args.lr)

    print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    if args.TM_flag == 1:
        faf_module.model.module.compensation.TM_Flag=True
    
    cal_epoch = args.cal_epoch
    for epoch in range(start_epoch, num_epochs + 1):
        if (epoch - cal_epoch) <= 24:
            for i in range(len(training_dataset.tau)):
                training_dataset.tau[i] = int((epoch - cal_epoch) / 3) + 1
            training_dataset.tau[center_agent] = 0
            faf_module.model.module.tau = training_dataset.tau
            training_dataset.get_historical_data_dict()

        elif (epoch - cal_epoch) > 36:
            latency = int(np.round(np.random.exponential(4,1)))
            for i in range(len(training_dataset.tau)):
                training_dataset.tau[i] = latency
                if training_dataset.tau[i] >= 10:
                    training_dataset.tau[i] = 10
                if training_dataset.tau[i] <= 0:
                    training_dataset.tau[i] = 1
            training_dataset.tau[center_agent] = 0
            training_dataset.get_historical_data_dict()
            faf_module.model.module.tau = training_dataset.tau
            faf_module.model.module.compensation.TM_Flag=True
            
        lr = faf_module.optimizer.param_groups[0]["lr"]
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter("Total loss", ":.6f")
        running_loss_class = AverageMeter(
            "classification Loss", ":.6f"
        )  # for cell classification error
        running_loss_loc = AverageMeter(
            "Localization Loss", ":.6f"
        )  # for state estimation error
        running_loss_clf = AverageMeter("Compensation loss f", ":.6f")
        running_loss_clh = AverageMeter("Compensation loss h", ":.6f")


        faf_module.model.train()
        faf_module.model.module.u_encoder.eval()
        faf_module.model.module.decoder.eval()
        faf_module.model.module.regression.eval()
        faf_module.model.module.classification.eval()
        faf_module.model.module.pixel_weighted_fusion.eval()

        t = tqdm(training_data_loader)
        for sample in t:
            (
                padded_voxel_point_list,  # voxelized point cloud for individual agent
                padded_voxel_points_teacher_list,  # fused voxelized point cloud for all agents (multi-view)
                label_one_hot_list,  # one hot labels
                reg_target_list,  # regression targets
                reg_loss_mask_list,
                anchors_map_list,  # anchor boxes
                vis_maps_list,
                target_agent_id_list,
                num_agent_list,  # e.g. 6 agent in current scene: [6,6,6,6,6,6], 5 agent in current scene: [5,5,5,5,5,0]
                trans_matrices_list_supervision, # trans_matrices to get Real-Time H
                trans_matrices_list,  # matrix for coordinate transformation. e.g. [batch_idx, j, i] ==> transformation matrix to transfer from agent i to j
                trans_matrices_newest_list, # trans to the newest frame of each agent
                padded_voxel_point_supervision_list #voxelized point cloud for individual agent to supervision the results of compensation
                # current_pose_rec_rotation_list, # rotation(4) for ego pose
                # current_pose_rec_translation_list, # translation(3) for ego pose
                # current_cs_rec_rotation_list, # rotation(4) for calibrated_sensor
                # current_cs_rec_translation_list # translation(3) for calibrated_sensor
    
            ) = zip(*sample)

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            trans_matrices_supervision = torch.stack(tuple(trans_matrices_list_supervision), 1)
            trans_matrices_newest = torch.stack(trans_matrices_newest_list, 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_all_agents = torch.stack(tuple(num_agent_list), 1)
            # current_pose_rec_rotation = torch.stack(tuple(current_pose_rec_rotation_list), 1)
            # current_pose_rec_translation = torch.stack(tuple(current_pose_rec_translation_list), 1)
            # current_cs_rec_rotation = torch.stack(tuple(current_cs_rec_rotation_list), 1)
            # current_cs_rec_translation = torch.stack(tuple(current_cs_rec_translation_list), 1)

            # add pose noise
            if pose_noise > 0:
                apply_pose_noise(pose_noise, trans_matrices)

            if not args.rsu:
                num_all_agents -= 1

            if flag == "upperbound":
                padded_voxel_point = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
            else:
                padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0) 
                padded_voxel_point_supervision = torch.cat(tuple(padded_voxel_point_supervision_list), 0)
            # label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            # reg_target = torch.cat(tuple(reg_target_list), 0)
            # reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            # anchors_map = torch.cat(tuple(anchors_map_list), 0)
            label_one_hot = label_one_hot_list[center_agent]
            reg_target = reg_target_list[center_agent]
            reg_loss_mask = reg_loss_mask_list[center_agent]
            anchors_map = anchors_map_list[center_agent]
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            data = {
                "bev_seq": padded_voxel_point.to(device),
                "labels": label_one_hot.to(device),
                "reg_targets": reg_target.to(device),
                "anchors": anchors_map.to(device),
                "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
                "vis_maps": vis_maps.to(device),
                "target_agent_ids": target_agent_id.to(device),
                "num_agent": num_all_agents.to(device),
                "trans_matrices_supervision": trans_matrices_supervision,
                "trans_matrices": trans_matrices,
                "trans_matrices_newest": trans_matrices_newest,
                "bev_supervision": padded_voxel_point_supervision
                # "current_pose_rec_rotation": current_pose_rec_rotation,
                # "current_pose_rec_translation": current_pose_rec_translation,
                # "current_cs_rec_rotation": current_cs_rec_rotation,
                # "current_cs_rec_translation": current_cs_rec_translation
            }

            if args.kd_flag == 1:
                padded_voxel_points_teacher = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
                data["bev_seq_teacher"] = padded_voxel_points_teacher.to(device)
                data["kd_weight"] = args.kd_weight

            loss, cls_loss, loc_loss, compensation_loss_f, compensation_loss_h = faf_module.step(
                data, batch_size, num_agent=num_agent
            )

            loss = loss.cpu().detach().numpy()
            cls_loss = cls_loss.cpu().detach().numpy()
            loc_loss = loc_loss.cpu().detach().numpy()
            compensation_loss_f = compensation_loss_f.cpu().detach().numpy()
            compensation_loss_h = compensation_loss_h.cpu().detach().numpy()
            running_loss_disp.update(loss)
            running_loss_class.update(cls_loss)
            running_loss_loc.update(loc_loss)
            running_loss_clf.update(compensation_loss_f)
            running_loss_clh.update(compensation_loss_h)



            if np.isnan(loss) or np.isnan(cls_loss) or np.isnan(loc_loss) or np.isnan(compensation_loss_f) or np.isnan(compensation_loss_h) :
                print(f"Epoch {epoch}, loss is nan: {loss}, {cls_loss} {loc_loss}")
                sys.exit()

            t.set_description("Epoch {},     lr {}".format(epoch, lr))
            t.set_postfix(
                cls_loss=running_loss_class.avg, loc_loss=running_loss_loc.avg,clf_loss=running_loss_clf.avg, clh_loss=running_loss_clh.avg
            )

        faf_module.scheduler.step()

        # save model
        if need_log:
            saver.write(
                "{}\t{}\t{}\t{}\t{}\n".format(
                    running_loss_disp, running_loss_class, running_loss_loc, running_loss_clf, running_loss_clh
                )
            )
            saver.flush()
            if config.MGDA:
                save_dict = {
                    "epoch": epoch,
                    "encoder_state_dict": faf_module.encoder.state_dict(),
                    "optimizer_encoder_state_dict": faf_module.optimizer_encoder.state_dict(),
                    "scheduler_encoder_state_dict": faf_module.scheduler_encoder.state_dict(),
                    "head_state_dict": faf_module.head.state_dict(),
                    "optimizer_head_state_dict": faf_module.optimizer_head.state_dict(),
                    "scheduler_head_state_dict": faf_module.scheduler_head.state_dict(),
                    "loss": running_loss_disp.avg,
                }
            else:
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": faf_module.model.state_dict(),
                    "optimizer_state_dict": faf_module.optimizer.state_dict(),
                    "scheduler_state_dict": faf_module.scheduler.state_dict(),
                    "loss": running_loss_disp.avg,
                }
            torch.save(
                save_dict, os.path.join(model_save_path, "epoch_" + str(epoch) + ".pth")
            )

    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default='/GPFS/public/V2XSim2LA/train',
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    parser.add_argument("--nepoch", default=200, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=0, type=int, help="Number of workers")
    parser.add_argument('--tau', default=[2,0,2,2,2,2], nargs='+', type=int, help='Latency for each agent')
    parser.add_argument('--untrainable_list', default=['decoder', 'encoder', 'classification', 'regression', 'fusion'], nargs='+', type=str, help='Latency for each agent')
    parser.add_argument("--k", default=3, type=int, help="How many frames do you want to use in compensation")
    parser.add_argument("--LA_Flag", default=True, type=bool, help="whether enable LA")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
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
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--warp_flag", default=0, type=int, help="Whether to use pose info for When2com"
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
        "--visualization", default=False, help="Visualize validation result"
    )
    parser.add_argument(
        "--com",
        default="syncdisco",
        type=str,
        help="lowerbound/upperbound/disco/when2com/v2v/sum/mean/max/cat/agent",
    )
    parser.add_argument(
        "--compensation_flag",
        default="SyncFormer",
        type=str,
        help="which compensation method to use",
    )
    parser.add_argument("--rsu", default=1, type=int, help="0: no RSU, 1: RSU")
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--center_agent", default=1, type=int, help="which agent is the center agent."
    )
    parser.add_argument(
        "--TM_flag", default=0, type=int, help="Whether to use Time Modulation."
    )
    parser.add_argument(
        "--cal_epoch", default=100, type=int, help="Whether to use Time Modulation."
    )
    parser.add_argument(
        "--auto_resume_path",
        default="./logs",
        type=str,
        help="The path to automatically reload the latest pth",
    )
    parser.add_argument(
        "--compress_level",
        default=2,
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
