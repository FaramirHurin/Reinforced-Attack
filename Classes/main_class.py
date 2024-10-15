import logging
import time
import pandas as pd
from datetime import datetime
from multiprocessing.pool import AsyncResult
from typing import Literal, ClassVar
from sklearn.neural_network import MLPClassifier
import rl
import warnings
import numpy as np
import polars as pl
import typed_argparse as tap
from sklearn.ensemble import RandomForestClassifier
import os
import util

from Classes.datset_inteface import Dataset
from quantile_forest import QuantileForest_Bandit
from rl.agents import PPO
from Baselines.baselines_classes import BaselineAgent
from fraud_env import FraudEnv
from datset_inteface import DatasetLoader
# Add the folder to sys.path
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '/rl'))
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '/Baselines'))
"""
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_dir = os.path.join(parent_dir, 'rl')
sys.path.append(target_dir)
target_dir = os.path.join(parent_dir, 'Baselines')
sys.path.append(target_dir)
pd.options.display.max_columns = None
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
dotenv.load_dotenv()
logging.getLogger().setLevel(os.getenv("LOG_LEVEL", "ERROR"))
logging.basicConfig(level=logging.ERROR)  # Only log ERROR level messages and above
"""
warnings.filterwarnings("ignore", message="X does not have valid feature names")
pd.set_option('display.max_columns', None)


class Args(tap.TypedArgs):
    logdir: str = tap.arg(help="Directory to save logs")
    fraud_balance: Literal["very-imbalanced", "imbalanced", "balanced"] = tap.arg(help="Fraud balance in synthetic datasets")
    k: int = tap.arg(help="Number of known features")
    u: int = tap.arg(help="Number of unknown features")
    n_steps: int = tap.arg(help="Number of steps to train the agent")
    challenge: Literal["fraudulent", "genuine", "all"] = tap.arg(help="Only use frauds in the environment observations")
    reward_type: Literal["probability", "label"] = tap.arg(help="Type of reward to use")
    run_num: int = tap.arg(help="Run number")
    min_values: np.array  = tap.arg(help="Min values features can have")
    max_values: np.array  =tap.arg(help="Max values features can have")

    @property
    def directory(self):
        return os.path.join(self.logdir,  f"k-{self.k}", f"u-{self.u}", self.challenge, f"reward-{self.reward_type}") #self.dataset,

def test_agent(
    env: FraudEnv,
    clf: RandomForestClassifier | MLPClassifier,
    agent: BaselineAgent,
    n_steps: int,
    reward_type: Literal["probability", "label"],
):
    actions = pl.DataFrame(agent.select_actions_batch(n_steps), env.actions)
    transactions = env.get_batch(n_steps)
    modified_transactions = transactions.with_columns([actions[col] for col in env.actions])
    if reward_type == "probability":
        rewards = clf.predict_proba(modified_transactions.to_numpy())[:, 0]
    elif reward_type == "label":
        preds = clf.predict(modified_transactions.to_numpy())
        rewards = np.ones(preds.shape) - preds
    else:
        print("Not the right type of reward")
        raise Exception
    return rewards


def experiment(args: Args, dataset: Dataset, clf: RandomForestClassifier | MLPClassifier):
    env = FraudEnv(
        transactions=dataset.env_transactions(args.challenge),
        k=args.k,
        u=args.u,
        classifier=clf,
        reward_type=args.reward_type,
        min_values=args.min_values,
        max_values=args.max_values
    )
    print('Number of frauds is ' + str(sum(dataset._test_y)))
    x_train = dataset.train_x
    percentiles = pd.DataFrame(np.percentile(x_train, [5, 95], axis=0),
                               columns=x_train.columns).loc[:, env.actions].values
    action_min = percentiles[0]
    action_max = percentiles[1]
    assert env.n_actions > 0

    bandit = QuantileForest_Bandit(action_min, action_max)
    ppo = PPO(
        env.observation_shape[0],
        env.n_actions,
        lr_actor=5e-4,
        gamma=0.9,
        lr_critic=5e-4,
        k_epochs=20,
        eps_clip=0.15,
    )
    logs = dict[str, np.ndarray]()
    logs["PPO"] = rl.train.train_agent(ppo, env, args.n_steps)
    # logs["Quantile Forest"] = rl.train.train_bandit(bandit, env, args.n_steps)

    mimicry_transactions = "genuine"
    for sampling in ("multivariate", "univariate", "uniform", "mixture"):
        for dataset_size in ("1k",  "100%"): # "5%",
            agent = BaselineAgent(
                k=args.k, test_x=dataset.mimicry_transactions(mimicry_transactions, dataset_size, args.u),
                generation_method=sampling
            )
            key = f"{sampling}-{mimicry_transactions}-{dataset_size}"
            logs[key] = test_agent(env, clf, agent, args.n_steps, args.reward_type)
            del agent

    os.makedirs(args.directory, exist_ok=True)
    filename = str(args.directory)+"/file.csv"
    pd.DataFrame(logs).to_csv(filename, index=False)
    print(pd.DataFrame(logs).describe())
    return filename


def run_all_experiments(dataset_types, n_features_list, clusters_list, class_sep_list, balance_list,
                    classifier_name, min_max_quantile, VARIABLES_STEPS, N_REPETITIONS, N_STEPS,
                    PROCESS_PER_GPU, N_GPUS):
    LOGDIR_OUTER = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(LOGDIR_OUTER, exist_ok=False)
    for dataset_type in dataset_types:

        LOGS_DATASET = os.path.join(LOGDIR_OUTER, dataset_type)
        os.makedirs(LOGS_DATASET, exist_ok=False)

        n_processes = N_GPUS * PROCESS_PER_GPU
        # pool = mp.Pool(n_processes)

        dataset_loader = DatasetLoader( dataset_type=dataset_type, classifier=classifier_name,
                                         n_features_list=n_features_list, clusters_list=clusters_list,
                                         class_sep_list=class_sep_list,balance_list=balance_list, )
        datasets, classifiers = dataset_loader.load()

        for key in datasets.keys():
            dataset = datasets[key]
            classifier = classifiers[key]
            for experiment_number in range(N_REPETITIONS):
                LOGDIR = os.path.join(LOGS_DATASET, "_" + str(experiment_number))
                if dataset_type == 'SkLearn':
                    folder_name = f"n_features={key[2]}_n_clusters={key[3]}_class_sep={key[4]}_balance={key[5]}"
                elif dataset_type == 'Kaggle':
                    folder_name = f"balance={key[2]}"
                elif dataset_type == 'Generator':
                    folder_name = f"balance={key[2]}"

                handles = [] #?
                save_path = f"{LOGDIR}/{folder_name}/{experiment_number}"

                # print('Save path is ' + str(save_path))
                print("Experiment number is " + str(experiment_number))

                os.makedirs(save_path, exist_ok=False)  # ?d

                n_actions_values = tuple(range(8, VARIABLES_STEPS -1, -VARIABLES_STEPS)) # -10
                for U_VALUE in [ 6]:
                    for K_VALUE in [ 4]: #, max(0, dataset._test_x.shape[1] - U_VALUE -3)
                        print('K = ' + str(K_VALUE) + ' U = ' + str(U_VALUE))
                        df_negative =  dataset.env_transactions("genuine")
                        min_values = df_negative.quantile(min_max_quantile)
                        max_values = df_negative.quantile(1 - min_max_quantile)
                        for reward_type in ("probability", "label"):
                            args = Args(logdir=save_path, k=K_VALUE, u = U_VALUE, n_steps=N_STEPS,
                            challenge="fraudulent", reward_type=reward_type, run_num=experiment_number,
                                        min_values=min_values, max_values=max_values)
                            handle = experiment(args, dataset, classifier)
                            handles.append((args, datetime.now(), handle))

        # Join here to avoid training all the classifiers while the experiments are running
        #join(LOGDIR, handles, experiment_number)


def join(logdir: str, handles: list[tuple[Args, datetime, AsyncResult]], experiment_number):
    experiments_logs = []
    while len(handles) > 0:
        i = 0
        while i < len(handles):
            print('We arrived at handles ' + str(i))
            args, start, path = handles[i]
            logging.info(f"Run {args.run_num} finished, {len(handles)} experiments remaining")
            experiments_logs.append({**args.__dict__, "start": start, "path": path}) # "end": end,
            # pl.DataFrame(experiments_logs).write_csv(f"{logdir}//experiments" + str(experiment_number) + ".csv")
            del handles[i]
            i += 1
        time.sleep(1)


UNDERSAMPLE = util.is_debugging()
dataset_types = [  'Kaggle','Generator', 'SkLearn']  # Kaggle  Generator SkLearn
n_features_list = [16, 32, 64]
clusters_list = [1, 8, 16]
class_sep_list = [0.5, 1, 2, 8]
balance_list = [0.1, 0.5]
classifier_name = 'RF'
min_max_quantile = 0.025
VARIABLES_STEPS = 2
N_REPETITIONS = 2 # 20
N_STEPS = 2000  # 0
PROCESS_PER_GPU = 2
N_GPUS = 1  # 8
run_all_experiments(dataset_types, n_features_list, clusters_list, class_sep_list, balance_list,
                    classifier_name, min_max_quantile, VARIABLES_STEPS, N_REPETITIONS, N_STEPS,
                    PROCESS_PER_GPU, N_GPUS)

# IMPOSE RANDOMIZATION