import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []
#初始化数据，从T，O，M表格中读取数据，取出一些无关数据并将这三个表格通过team树形列进行连接：
#根据每个队伍的Miscellaneous Opponent，Team统计数据csv文件进行初始化
def initialize_data(Mstat,Ostat,Tstat):
    new_Mstat = Mstat.drop(['Rk','Arena'],axis=1)
    new_Ostat = Ostat.drop(['Rk',"G",'MP'],axis=1)
    new_Tstat = Tstat.drop(['Rk',"G",'MP'],axis=1)
    team_stats1 = pd.merge(new_Mstat,new_Ostat,how='left',on='Team')
    team_stats1 = pd.merge(team_stats1,new_Tstat,how='left',on='Team')
    return team_stats1.set_index('Team',inplace=False,drop=True)
def get_elo(team):
    try:
        return team_elos[team]
    except:
        team_elos[team] = base_elo
    return team_elos[team]
def calc_elo(win_team,lose_team):
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)
    #根据Logistic Distribution计算 PK 双方（A和B）对各自的胜率期望值计算公式
    rank_diff = winner_rank - loser_rank
    exp = (rank_diff *-1)/400
    odds  = 1/(1+math.pow(10,exp))
    #根据rank界别修改k值
    if winner_rank < 2100:
        k = 32
    elif winner_rank >=2100 and winner_rank <2400:
        k = 24
    else:
        k=16
    #更新rank数值
    new_winner_rank = round(winner_rank+(k*(1-odds)))
    new_loser_rank = round(loser_rank+(k*(0-odds)))
    return new_winner_rank,new_loser_rank
 
#基于统计好的数据，给每只队伍的eloscore计算结果，建立对应15-16年数据集，我们认为主场作战的队伍更有优势，因此会给主场队伍加上100分
def build_dataSet(all_data):
    print("Building data set..")
    X = []
    skip = 0
    for index,row in all_data.iterrows():
        Wteam = row['WTeam']
        Lteam = row['LTeam']
        #获取最初的elo或者每个队伍最初的elo值
        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)
        #给主场比赛队伍加上100的elo值
        if row['WLoc'] == 'H':
            team1_elo += 100
        else:
            team2_elo += 100
        #把elo当成评价每个队伍的第一个特征值
        team1_features = [team1_elo]
        team2_features = [team2_elo]
        # 添加我们从basketball reference.com获得的每个队伍的统计信息
        for key,value in team_stats.loc[Wteam].iteritems():
            team1_features.append(value)
        for key,value in team_stats.loc[Lteam].iteritems():
            team2_features.append(value)
        # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
        # 并将对应的0/1赋给y值
        if random.random() > 0.5:
            X.append(team1_features+team2_features)
            y.append(0)
        else:
            X.append(team2_features+team1_features)
            y.append(1)
        if skip ==0:
            print('X',X)
            skip = 1
        new_winner_rank,new_loser_rank = calc_elo(Wteam,Lteam)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank
    return np.nan_to_num(X),y
#最终利用训练好的模型在 16~17 年的常规赛数据中进行预测
def predict_winner(team_1, team_2, model):
    features = []
    # team 1，客场队伍
    features.append(get_elo(team_1))
    for key, value in team_stats.loc[team_1].iteritems():
        features.append(value)
    # team 2，主场队伍
    features.append(get_elo(team_2) + 100)
    for key, value in team_stats.loc[team_2].iteritems():
        features.append(value)
    features = np.nan_to_num(features)
    return model.predict_proba([features])
#最终在 main 函数中调用这些数据处理函数，使用 sklearn 的Logistic Regression方法建立回归模型
if __name__=='__main__':
    folder = 'data'
    Mstat = pd.read_csv(folder + '/15-16Miscellaneous_Stat.csv')
    Ostat = pd.read_csv(folder + '/15-16Opponent_Per_Game_Stat.csv')
    Tstat = pd.read_csv(folder + '/15-16Team_Per_Game_Stat.csv')
    team_stats = initialize_data(Mstat, Ostat, Tstat)
    result_data = pd.read_csv(folder + '/2015-2016_result.csv')
    X, y = build_dataSet(result_data)
    #训练网络模型
    print("Fitting on %d game samples.." % len(X))
    model = linear_model.LogisticRegression()
    model.fit(X,y)
    print("Doing cross-validation..")
    cross_val_score(model,X,y,cv = 10,scoring='accuracy',n_jobs=-1).mean()
    print(model)
    print('Predicting on new schedule..')
    schedule1617 = pd.read_csv(folder + '/16-17Schedule.csv')
    result = []
    for index, row in schedule1617.iterrows():
        team1 = row['Vteam']
        team2 = row['Hteam']
        pred = predict_winner(team1, team2, model)
        prob = pred[0][0]
        if prob > 0.5:
            winner = team1
            loser = team2
            result.append([winner, loser, prob])
        else:
            winner = team2
            loser = team1
            result.append([winner, loser, 1 - prob])
    with open('16-17Result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['win', 'lose', 'probability'])
        writer.writerows(result)
        print('done.')