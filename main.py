import torch
import shutil

from sklearn.model_selection import ShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from ae_model.preprocessing import Load, CreateLoader, VacDataset
from ae_model.autoencoder import AEModel, TrainAE,
from ae_model.prediction import Config, Loss, Predict

if __name__ == '__main__':
    ### Configure file names and device
    train_data_filename = '../input/stanford-covid-vaccine/train.json'
    test_data_filename = '../input/stanford-covid-vaccine/test.json'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Load and preprocess training data and test data to feed into denoising Auto Encoder (AE) model
    load = Load(base_train_data=train_data_filename, base_test_data=test_data_filename)
    denoised_train_data = load.denoise()
    public_df, private_df = load.query_seq_length()
    features, _ = load.preprocess(denoised_train_data, True)
    features_tensor = torch.from_numpy(features)

    create_loader = CreateLoader()
    structure_adj0 = create_loader.get_structure_adj(denoised_train_data)
    distance_matrix0 = create_loader.get_distance_matrix(structure_adj0.shape[1])
    dataset0 = VacDataset(features_tensor, denoised_train_data, structure_adj0, distance_matrix0, None)
    features, _ = load.preprocess(public_df, True)
    features_tensor = torch.from_numpy(features)

    structure_adj1 = create_loader.get_structure_adj(public_df)
    distance_matrix1 = create_loader.get_distance_matrix(structure_adj1.shape[1])
    dataset1 = VacDataset(features_tensor, public_df, structure_adj1, distance_matrix1, None)
    features, _ = load.preprocess(private_df, True)
    features_tensor = torch.from_numpy(features)

    structure_adj2 = create_loader.get_structure_adj(private_df)
    distance_matrix2 = create_loader.get_distance_matrix(structure_adj2.shape[1])
    dataset2 = VacDataset(features_tensor, private_df, structure_adj2, distance_matrix2, None)

    loader0 = torch.utils.data.DataLoader(dataset0, batch_size=64, shuffle=False, drop_last=False)
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=False, drop_last=False)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=False, drop_last=False)

    ### Pre-train denoising AE model
    ae_model = AEModel()
    ae_model.to(device)
    train_ae_model = TrainAE(ae_model)
    train_ae_model = TrainAE(ae_model)
    res = dict(end_epoch=0, it=0, min_loss_epoch=0)
    epochs = [2, 2, 2, 2] #[5, 5, 5, 5]
    for e in epochs:
        res = train_ae_model.train_ae(loader0, e, device=device)
        res = train_ae_model.train_ae(loader1, e, device=device)
        res = train_ae_model.train_ae(loader2, e, device=device)

    epoch = res["min_loss_epoch"]
    shutil.copyfile(f"./model/model-{epoch}.pt", "ae-model.pt")

    ### Train regression model from pre-trained AE model
    cfg = Config()
    loss_eval = Loss()

    split = ShuffleSplit(n_splits=cfg.k_folds, test_size=cfg.test_size)
    ids = load.base_train_data.reset_index()['id']
    save_path = Path("./model_prediction")
    if not save_path.exists():
        save_path.mkdir()

    for fold, (train_index, test_index) in enumerate(split.split(load.base_train_data)):
        print(f"fold: {fold}")
        train_df = load.base_train_data.loc[train_index].reset_index()
        val_df = load.base_train_data.loc[test_index].reset_index()

        # K-th fold training data
        train_features, train_labels = load.preprocess(train_df)
        train_features_tensor = torch.from_numpy(train_features)
        train_labels_tensor = torch.from_numpy(train_labels)
        train_structure_adj = create_loader.get_structure_adj(train_df)
        train_distance_matrix =create_loader.get_distance_matrix(train_structure_adj.shape[1])
        train_dataset = VacDataset(train_features_tensor, train_df, train_structure_adj,
                                   train_distance_matrix, train_labels_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, cfg.BATCH_SIZE, shuffle=True, drop_last=False)

        # K-th fold validation data
        valid_features, valid_labels = load.preprocess(val_df)
        valid_features_tensor = torch.from_numpy(valid_features)
        valid_labels_tensor = torch.from_numpy(valid_labels)
        valid_structure_adj = create_loader.get_structure_adj(val_df)
        valid_distance_matrix =create_loader.get_distance_matrix(valid_structure_adj.shape[1])
        valid_dataset = VacDataset(valid_features_tensor, val_df, valid_structure_adj,
                                   valid_distance_matrix, valid_labels_tensor)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, cfg.BATCH_SIZE, shuffle=True, drop_last=False)

        ae_model = AEModel()
        state_dict = torch.load("./ae-model.pt")
        ae_model.load_state_dict(state_dict)
        del state_dict

        model = FromAeModel(ae_model.seq)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train regression model
        losses = []
        writer = SummaryWriter(f"logs/{fold}")
        it = 0
        model_save_path = Path("./model")
        start_epoch = 0
        end_epoch = start_epoch + cfg.epoch
        if not model_save_path.exists():
            model_save_path.mkdir(parents=True)
        min_eval_loss = 10.0
        min_eval_epoch = None
        for epoch in progress_bar(range(start_epoch, end_epoch)):
            print(f"epoch: {epoch}")
            model.train()
            for i, data in enumerate(train_loader):
                _, loss = loss_eval.learn_from_batch(model, data, optimizer, cfg.lr_scheduler, cfg.device)
                loss_v = loss.item()
                writer.add_scalar('loss', loss_v, it)
                losses.append(loss_v)
                it += 1
            print(f'epoch: {epoch} loss: {np.mean(losses)}')
            losses = []

            eval_result = loss_eval.evaluate(model, valid_loader, cfg.device)
            eval_loss = eval_result["loss"]
            if eval_loss <= min_eval_loss:
                min_eval_epoch = epoch
                min_eval_loss = eval_loss

            print(f"eval loss: {eval_loss} {eval_result['mcmse']}")
            writer.add_scalar(f"evaluate/loss", eval_loss, epoch)
            writer.add_scalar(f"evaluate/mcmse", eval_result["mcmse"], epoch)
            model.train()
            torch.save(optimizer.state_dict(), str(model_save_path / "optimizer.pt"))
            torch.save(model.state_dict(), str(model_save_path / f"model-{epoch}.pt"))

        print(f'min eval loss: {min_eval_loss} epoch {min_eval_epoch}')

        shutil.copyfile(f"./model/model-{min_eval_epoch}.pt", f"model_prediction/model-{fold}.pt")
        del model


    ### Generate predictions from trained regression model
    # Load and preprocess public test data to feed into regression model
    pub_features, pub_labels = load.preprocess(public_df, is_test=True)
    pub_features_tensor = torch.from_numpy(pub_features)
    pub_labels_tensor = torch.from_numpy(pub_labels)
    pub_structure_adj = create_loader.get_structure_adj(public_df)
    pub_distance_matrix = create_loader.get_distance_matrix(pub_structure_adj.shape[1])
    pub_dataset = VacDataset(pub_features_tensor, public_df, pub_structure_adj,
                             pub_distance_matrix, pub_labels_tensor)
    pub_loader = torch.utils.data.DataLoader(pub_dataset, 1, shuffle=True, drop_last=False)

    # Load and preprocess private test data to feed into regression model
    pri_features, pri_labels = load.preprocess(private_df, is_test=True)
    pri_features_tensor = torch.from_numpy(pri_features)
    pri_labels_tensor = torch.from_numpy(pri_labels)
    pri_structure_adj = create_loader.get_structure_adj(private_df)
    pri_distance_matrix = create_loader.get_distance_matrix(pri_structure_adj.shape[1])
    pri_dataset = VacDataset(pri_features_tensor, private_df, pri_structure_adj,
                             pri_distance_matrix, pri_labels_tensor)
    pri_loader = torch.utils.data.DataLoader(pri_dataset, 1, shuffle=True, drop_last=False)

    # Predict results for public and private test data
    predict = Predict()
    pred_df_list = []
    c = 0
    for fold in range(cfg.k_folds):
        model_load_path = f"./model_prediction/model-{fold}.pt"
        ae_model0 = AEModel()
        ae_model1 = AEModel()
        model_pub = FromAeModel(pred_len=107, seq=ae_model0.seq)
        model_pub = model_pub.to(device)
        model_pri = FromAeModel(pred_len=130, seq=ae_model1.seq)
        model_pri = model_pri.to(device)
        state_dict = torch.load(model_load_path, map_location=device)
        model_pub.load_state_dict(state_dict)
        model_pri.load_state_dict(state_dict)
        del state_dict

        data_list = []
        data_list += predict.predict_data(model_pub, pub_loader, cfg.device, 1, load.target_cols)
        data_list += predict.predict_data(model_pri, pri_loader, cfg.device, 1, load.target_cols)
        pred_df = pd.DataFrame(data_list, columns=["id_seqpos"] + load.target_cols)
        pred_df_list.append(pred_df)
        c += 1

    data_dic = dict(id_seqpos=pred_df_list[0]["id_seqpos"])
    for col in load.target_cols:
        vals = np.zeros(pred_df_list[0][col].shape[0])
        for df in pred_df_list:
            vals += df[col].values
        data_dic[col] = vals / float(c)
    pred_df_avg = pd.DataFrame(data_dic, columns=["id_seqpos"] + load.target_cols)
    pred_df_avg.to_csv("./submission.csv", index=False)
