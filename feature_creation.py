import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
import db_utils
import argparse

config_dict = {
        'ma_periods':[1,3,5],
        'team_stats':[{'home_stat_name':'home_goals_scored',
                      'away_stat_name':'away_goals_scored',
                      'new_val_name':'goals_scored'},
                      {'home_stat_name':'away_goals_scored',
                       'away_stat_name':'home_goals_scored',
                       'new_val_name':'goals_conceded'},
                      {'home_stat_name':'home_shots',
                       'away_stat_name':'away_shots',
                       'new_val_name':'shots_taken'},
                      {'home_stat_name':'home_shots_target',
                       'away_stat_name':'away_shots_target',
                       'new_val_name':'shots_taken_on_target'},
                      {'home_stat_name':'away_shots',
                       'away_stat_name':'home_shots',
                       'new_val_name':'shots_allowed'},
                      {'home_stat_name':'away_shots_target',
                       'away_stat_name':'home_shots_target',
                       'new_val_name':'shots_allowed_on_target'},
                      {'home_stat_name':'home_corners_won',
                       'away_stat_name':'away_corners_won',
                       'new_val_name':'corners_won'},
                      {'home_stat_name':'away_corners_won',
                       'away_stat_name':'home_corners_won',
                       'new_val_name':'corners_conceded'}
                    ]
    }


model_inputs = [
        'points_to_date','pct_points_won_to_date','pct_points_won_ma1',
        'pct_points_won_ma3','pct_points_won_ma5','goals_scored_mean_to_date',
        'goals_scored_ma1','goals_scored_ma3','goals_scored_ma5','goals_conceded_mean_to_date',
        'goals_conceded_ma1','goals_conceded_ma3','goals_conceded_ma5','shots_taken_mean_to_date',
        'shots_taken_ma1','shots_taken_ma3','shots_taken_ma5','shots_taken_on_target_mean_to_date',
        'shots_taken_on_target_ma1','shots_taken_on_target_ma3','shots_taken_on_target_ma5',
        'shots_allowed_mean_to_date','shots_allowed_ma1','shots_allowed_ma3','shots_allowed_ma5',
        'shots_allowed_on_target_mean_to_date','shots_allowed_on_target_ma1',
        'shots_allowed_on_target_ma3','shots_allowed_on_target_ma5','corners_won_mean_to_date',
        'corners_won_ma1','corners_won_ma3','corners_won_ma5','corners_conceded_mean_to_date',
        'corners_conceded_ma1','corners_conceded_ma3','corners_conceded_ma5',
        'off_efficiency_mean_to_date','def_efficiency_mean_to_date','off_efficiency_ma1',
        'def_efficiency_ma1','off_efficiency_ma3','def_efficiency_ma3','off_efficiency_ma5',
        'def_efficiency_ma5','prior_elo_rank_ma1','prior_elo_rank_ma3','prior_elo_rank_ma5'
    ]

def get_all_team_df(user,db_pwd):
    SQL = """
    SELECT * FROM games
    """
    eng = "postgresql+psycopg2://%s:%s@localhost:5432/soccer"%(user,db_pwd)
    engine = create_engine(eng)
    conn = engine.connect()
    all_teams_df = pd.read_sql(sql=SQL, con=engine)
    return all_teams_df


def column_cleanup(df):
    cols = df.columns
    new_columns = []
    for col in cols:
        if col.endswith('_x'):
            new_columns.append(col[:-2])
        elif col.endswith('_y'):
            df = df.drop([col], axis=1)
        else:
            new_columns.append(col)
    df.columns = new_columns
    return df


def make_team_df(all_teams_df,team_name):
    team_df = all_teams_df[(all_teams_df['home_team_name']==team_name) | \
                           (all_teams_df['away_team_name']==team_name)].copy()
    team_df['team_outcome'] = np.where(
            (team_df['home_team_name']==team_name) & (team_df['result']==0),3,
            np.where(
                (team_df['home_team_name']==team_name) & (team_df['result']==2),0,
                np.where(
                    (team_df['away_team_name']==team_name) & (team_df['result']==0),0,
                    np.where(
                        (team_df['away_team_name']==team_name) & (team_df['result']==2),3,
                        1
                        )
                    )
                )
            )
    team_df.index = team_df['game_id']
    points = team_df.groupby(['season'])['team_outcome'].cumsum().reset_index()
    points = pd.merge(points,team_df.loc[:,['game_id','season']].copy(),on='game_id')
    points.index = points['game_id']
    points = points.groupby('season')['team_outcome'].shift(1).reset_index()
    points.columns = ['game_id','points_to_date']
    team_df = pd.merge(team_df,points,on='game_id')
    return team_df


def add_points_won(team_df,ma_periods):
    team_df['pct_points_won_to_date'] = team_df['points_to_date'] / (team_df.groupby('season')['game_id'].cumcount()*3)
    for m in ma_periods:
        val_name = 'pct_points_won_ma%s'%(str(m))
        team_df[val_name] = team_df.groupby('season')['pct_points_won_to_date'].transform(lambda x: x.rolling(m).mean())
    return team_df


def add_team_stat(team_df,
                  team_name,
                  home_stat_name,
                  away_stat_name,
                  new_val_name,
                  ma_periods):
    values = np.where(team_df['home_team_name']==team_name,
                      team_df[home_stat_name],
                      team_df[away_stat_name])
    val_df = pd.DataFrame({'game_id':team_df['game_id'],
                           'season':team_df['season'],
                           'values':values})
    val_df.index = val_df['game_id']
    val_df = val_df.groupby('season')['values'].shift(1).reset_index()
    val_df = pd.merge(val_df,team_df.loc[:,['season','game_id']],on='game_id')
    val_df[new_val_name + '_mean_to_date'] = val_df.groupby('season')['values']. \
    transform(lambda x: x.expanding().mean())
    for m in ma_periods:
        val_df[new_val_name + '_ma%s'%(m)] = val_df.groupby('season')['values']. \
        transform(lambda x: x.rolling(m).mean())
    team_df = pd.merge(team_df,val_df,on=['game_id'])
    team_df = column_cleanup(team_df)
    return team_df


def off_def_efficiency(team_df,ma_periods):
    team_df['off_efficiency_mean_to_date'] = team_df['goals_scored_mean_to_date']/team_df['shots_taken_mean_to_date']
    team_df['def_efficiency_mean_to_date'] = team_df['goals_conceded_mean_to_date']/team_df['shots_allowed_mean_to_date']
    for ma in ma_periods:
        team_df['off_efficiency_ma%s'%(str(ma))] = team_df['goals_scored_ma%s'%(str(ma))]/team_df['shots_taken_ma%s'%(str(ma))]
        team_df['def_efficiency_ma%s'%(str(ma))] = team_df['goals_conceded_ma%s'%(str(ma))]/team_df['shots_allowed_ma%s'%(str(ma))]
    return team_df


def lag_elo(team_df,
            team_name,
            ma_periods):
    home_stat_name = 'home_elo_rank'
    away_stat_name = 'away_elo_rank'
    new_val_name = 'prior_elo_rank'
    values = np.where(team_df['home_team_name']==team_name,
                      team_df[home_stat_name],
                      team_df[away_stat_name])
    val_df = pd.DataFrame({'game_id':team_df['game_id'],
                           'season':team_df['season'],
                           'values':values})
    val_df.index = val_df['game_id']
    val_df['values'] = val_df['values'].shift(1)
    for m in ma_periods:
        val_df[new_val_name + '_ma%s'%(m)] = val_df['values'].rolling(m).mean()
    team_df = pd.merge(team_df,val_df,on=['game_id'])
    team_df = column_cleanup(team_df)
    return team_df


def add_all_stats(user,db_pwd,config_dict):
    all_teams_df = get_all_team_df(user=user, db_pwd=db_pwd)
    team_dfs = []
    for team in all_teams_df['home_team_name'].unique():
        team_df = make_team_df(all_teams_df, team)
        team_df = add_points_won(team_df, ma_periods=config_dict['ma_periods'])
        for stat_dict in config_dict['team_stats']:
            team_df = add_team_stat(team_df,
                                     team_name=team,
                                     home_stat_name=stat_dict['home_stat_name'],
                                     away_stat_name=stat_dict['away_stat_name'],
                                     new_val_name=stat_dict['new_val_name'],
                                     ma_periods=config_dict['ma_periods']
                                    )
        team_df = off_def_efficiency(team_df,config_dict['ma_periods'])
        team_df = lag_elo(team_df, team, config_dict['ma_periods'])
        team_df['target_team'] = team
        team_dfs.append(team_df)
    team_dfs = pd.concat(team_dfs)
    team_dfs.index = team_dfs['game_id']
    return team_dfs


def add_model_features(teams_df,user,db_pwd):
    all_game_ids = teams_df['game_id'].unique()
    model_input_dfs = []
    all_game_ids.sort()
    for game_id in all_game_ids:
        df = teams_df[teams_df['game_id']==game_id]
        home_df = df[df['home_team_name']==df['target_team']]
        away_df = df[df['away_team_name']==df['target_team']]
        base_df = home_df.loc[:,['game_id',
                                 'home_team_name',
                                 'away_team_name',
                                 'home_avg_win_odds',
                                 'draw_avg_win_odds',
                                 'away_avg_win_odds',
                                 'home_max_win_odds',
                                 'draw_max_win_odds',
                                 'away_max_win_odds',]
                              ]
        home_df = home_df.loc[:,model_inputs]
        away_df = away_df.loc[:,model_inputs]
        home_df.columns = ['home_' + str(col) for col in home_df.columns]
        away_df.columns = ['away_' + str(col) for col in away_df.columns]
        model_df = pd.concat([base_df,away_df,home_df],axis=1)
        model_input_dfs.append(model_df)

    model_df = pd.concat(model_input_dfs)
    model_df.index = model_df['game_id']
    model_df = model_df.sort_index()
    eng = "postgresql+psycopg2://%s:%s@localhost:5432/soccer"%(user,db_pwd)
    engine = create_engine(eng)
    conn = engine.connect()
    model_df = model_df.replace({np.inf:np.nan})
    db_utils.to_pg(model_df, 'model_inputs', engine)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='add model features to db')
    parser.add_argument('-p', '--db_pwd', type=str, required=True,
                        help='password for db')
    parser.add_argument('-u', '--user', type=str, required=True,
                        help='user for db')
    args = parser.parse_args()
    teams_df = add_all_stats(user=args.user,db_pwd=args.db_pwd,config_dict=config_dict)
    add_model_features(teams_df=teams_df,user=args.user,db_pwd=args.db_pwd)
