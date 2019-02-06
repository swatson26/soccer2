import pandas as pd
import numpy as np
from datetime import datetime
import datetime
from datetime import timedelta
import requests
# from pandas.tseries.offsets import *
import os
from StringIO import StringIO
import csv
import argparse
from sqlalchemy.engine import create_engine
import db_utils

LEAGUES = ['Premier', 'Championship', 'SerieA',
           'laLiga1', 'Bundesliga1', 'Championnat']

HIST_DATA_PATHS = {
        'Premier':'E0',
        'Championship':'E1',
        'Bundesliga1':'D1',
        'SerieA':'I1',
        'Championnat':'F1',
        'laLiga1':'SP1',
        }

BASE_PATH = 'http://www.football-data.co.uk/mmz4281/'

def download_data(data_folder, year_start_min, year_start_max):
    for l,ab in HIST_DATA_PATHS.iteritems():
        for start in range(year_start_min,year_start_max):
            if len(str(start)) == 1:
                date1 = '0'+ str(start)
            else:
                date1 = str(start)
            if len(str(start+1)) == 1:
                date2 = '0'+ str(start+1)
            else:
                date2 = str(start+1)
            date = date1 + date2
            path = BASE_PATH + date + '/' + ab +'.csv'
            df = pd.read_csv(path)
            if not os.path.exists(os.path.join(data_folder,l)):
                os.makedirs(os.path.join(data_folder,l))
            df.to_csv(os.path.join(data_folder,l,date+'_'+l+'.csv'))


def agg_raw_data(data_folder, leagues,year_start_min,
                 year_start_max):
    """
    After downloading a bunch of csvs from:
    http://www.football-data.co.uk/data.php, \
    this will join them all into single dataframes for each league

    Parameters:
    soccer_drive: string path to folder
    data_folder: path to raw data folder
    leagues: list of league names

    Output:
    csv and pickle datasets for every league
    """
    download_data(data_folder, year_start_min, year_start_max)
    colNames = ['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG', \
              'HTR','Referee','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR','Bb1X2','BbMxH',\
              'BbAvH','BbMxD','BbAvD','BbMxA','BbAvA','BbOU']
    for league in leagues:
        dfFinal = pd.DataFrame()
        print 'base league done: %s' %league
        for filename in os.listdir(os.path.join(data_folder ,league)):
            if filename.endswith('.csv'):
                if not filename.startswith('.'):
                    if not filename.startswith('Ico'):
                        if not filename.split('.')[0] in leagues:
                            dfraw = pd.read_csv(os.path.join(data_folder,league, filename))
                            df = dfraw.loc[:, colNames]
                            df['fileSource'] = filename
                            dfFinal = dfFinal.append(df)
        dfFinal['Date'] = pd.to_datetime(dfFinal['Date'], dayfirst=True)
        dfFinal.index = dfFinal['Date']
        dfFinal = dfFinal.sort_values('Date')
        dfFinal.to_csv(os.path.join(data_folder,league, league + '.csv'))
        dfFinal.to_pickle(os.path.join(data_folder,league, league + '.pkl'))


def add_elo_alt_name(league, data_folder):
    #TODO add error handling here
    df = pd.read_pickle(os.path.join(data_folder,league, league + '.pkl'))
    bundesliga_replace = {'Bayern Munich' : 'Bayern', 'FC Koln' :'Koeln', 'Werder Bremen':'Werder', \
                    'Ein Frankfurt': 'Frankfurt', 'Schalke 04': 'Schalke', 'Nurnberg': 'Nuernberg',\
                    "M'gladbach": 'Gladbach' ,'Fortuna Dusseldorf': 'Duesseldorf',
                    'Greuther Furth':'Fuerth',' Hansa Rostock':'Rostock',
                    'Kaiserslautern':'Lautern', }
    LaLiga_replace = {'Ath Bilbao' : 'Bilbao', 'Ath Madrid': 'Atletico', 'Espanol':'Espanyol', \
                      'La Coruna' : 'Depor', 'Sp Gijon': 'Gijon', 'Vallecano' : 'Rayo Vallecano',
                      'Gimnastic':'Tarragona'}
    Championnat_replace = {'St Etienne':'Saint-Etienne',
                           'Evian Thonon Gaillard':'Evian TG',
                           'Ajaccio GFCO':'Ajaccio',
                           'Areles':'Arles-Avignon'}
    Championship_replace = {"Nott'm Forest":"Forest",'Milton Keynes Dons':'MKDons',}
    Premier_replace = {"Nott'm Forest":"Forest",'Milton Keynes Dons':'MKDons',}

    replaces={'laLiga1':LaLiga_replace,
              'Bundesliga1':bundesliga_replace,
              'Championnat':Championnat_replace,
              'Championship':Championship_replace,
              'Premier':Premier_replace}
    try:
        e=df.replace({"HomeTeam": replaces[league]})
        df['EloNameHome']=e['HomeTeam']
        f=df.replace({"AwayTeam": replaces[league]})
        df['EloNameAway']=f['AwayTeam']
        df.to_csv(os.path.join(data_folder,league,league + '.csv'))
        df.to_pickle(os.path.join(data_folder,league,league + '.pkl'))
    except:
        df['EloNameAway']=df['AwayTeam']
        df['EloNameHome']=df['HomeTeam']
        df.to_csv(os.path.join(data_folder,league,league + '.csv'))
        df.to_pickle(os.path.join(data_folder,league,league + '.pkl'))
    return df


def get_elo_league(league, data_folder):
    """
    gets full elo history of every team in league, returns long format dataframe

    parameters:
    -----------
    league : sting of league name
    """
    df = pd.read_pickle(os.path.join(data_folder,league,league + '.pkl'))
    allTeams = list(df['EloNameHome'].value_counts().index)
    fullData=[]
    for team in allTeams:
        try:
            url="http://api.clubelo.com/"
            response = requests.get(url + team.replace(" ", "") )
            Data = StringIO(response.text)
            df1 = pd.read_csv(Data, sep=",")
            df1['From'] = pd.to_datetime(df1['From'])
            df1.index = df1['From']
            df1 = df1.sort_index()
            df1['Rank'] = np.where(df1['Rank']=='None', np.nan, df1['Rank'] )
            # reindex to have daily data, via front filling. API returns ts at irregular frequencies
            idx = pd.date_range(df1.index[0],df1.index.max())
            df2 = df1.reindex(idx, fill_value = np.nan)
            df2 = df2.fillna(method = 'ffill')
            df2['Date'] = df2.index
            df2 = df2.drop(['Country', 'Level', 'From', 'To'], axis=1)
            fullData.append(df2)
        except:
            print 'failed: %s'%(team)
            print url + team.replace(" ", "")
    fullDf = pd.concat(fullData, axis=0)
    return fullDf


def add_elo(elo_df, league, data_folder):
    """
    adds elo rankings to final league dataframe

    parameters:
    -----------
    elo_df : df from get_elo_league
    league_name : string of league name
    """
    league_df = pd.read_pickle(os.path.join(data_folder,league,league + '.pkl'))
    elo_df['EloNameHome'] = elo_df['Club']
    elo_df['EloNameAway'] = elo_df['Club']
    # if there is existing elo data, remove it
    try:
        league_df.drop(['RankHome', 'EloHome','RankAway','EloAway'], axis=1, inplace=True)
    except:
        pass
    # add home team elo
    league_df = pd.merge(league_df, elo_df, on=['Date', 'EloNameHome'], how='left', suffixes=("", '_drop'))
    league_df = league_df.drop(['Club', 'EloNameAway_drop'], axis=1)
    league_df = league_df.rename(columns={'Rank' : 'RankHome', 'Elo' : 'EloHome'})
    # add away team elo
    league_df = pd.merge(league_df, elo_df, on=['Date', 'EloNameAway'], how='left', suffixes=('', '_drop'))
    league_df = league_df.drop(['Club', 'EloNameHome_drop'], axis=1)
    league_df = league_df.rename(columns={'Rank' : 'RankAway', 'Elo' : 'EloAway'})
    return league_df


def elo_loop(leagues, data_folder):
    for league in leagues:
        elo_df = add_elo_alt_name(league, data_folder)
        elo_df = get_elo_league(league, data_folder)
        league_df = add_elo(elo_df, league, data_folder)
        league_df.to_csv(os.path.join(data_folder,league,league + '.csv'))
        league_df.to_pickle(os.path.join(data_folder,league,league + '.pkl'))
        print 'elo league complete: %s' %league


def populate_historical_data(data_folder,db_pwd,user):
    """

    """
    leagues = LEAGUES
    full_dfs = []
    for root, dirs, files in os.walk(data_folder):
        for fn in files:
            if fn.split('.')[0] in leagues and fn.endswith('.csv'):
                df_ = pd.read_csv(os.path.join(root, fn))
                df_['league'] = fn.split('.')[0]
                full_dfs.append(df_)
    full_df = pd.concat(full_dfs)
    full_df['Date'] = pd.to_datetime(full_df['Date'] )
    full_df.index = full_df['Date']
    full_df = full_df[~full_df['HomeTeam'].isnull()]
    # make season column
    max_dates = full_df.groupby('fileSource')['Date'].max().reset_index()
    max_dates['max_year'] = max_dates['Date'].dt.year
    min_dates = full_df.groupby('fileSource')['Date'].min().reset_index()
    min_dates['min_year'] = min_dates['Date'].dt.year
    season_dates = pd.merge(max_dates, min_dates, on='fileSource')
    season_dates['season'] = season_dates['min_year'].astype(str) + '-' +  season_dates['max_year'].astype(str)
    full_df = pd.merge(full_df, season_dates.loc[:,['fileSource', 'season']],
                       on='fileSource', how='left')
    full_df = full_df.rename(columns={   'Date':'game_date',
                                         'HomeTeam':'home_team',
                                         'AwayTeam':'away_team',
                                         'FTHG':'home_goals_scored',
                                         'FTAG':'away_goals_scored',
                                         'FTR':'result',
                                         'HS':'home_shots',
                                         'AS':'away_shots',
                                         'HST':'home_shots_target',
                                         'AST':'away_shots_target',
                                         'HF':'home_fouls_committed',
                                         'AF':'away_fouls_committed',
                                         'HC':'home_corners_won',
                                         'AC':'away_corners_won',
                                         'HY':'home_yellows_conceded',
                                         'AY':'away_yellows_conceded',
                                         'HR':'home_reds_conceded',
                                         'AR':'away_reds_conceded',
                                         'BbAvH':'home_avg_win_odds',
                                         'BbAvD':'draw_avg_win_odds',
                                         'BbAvA':'away_avg_win_odds',
                                         'BbMxH':'home_max_win_odds',
                                         'BbMxD':'draw_max_win_odds',
                                         'BbMxA':'away_max_win_odds',
                                         'AvgH':'home_avg_win_odds2',
                                         'AvgD':'draw_avg_win_odds2',
                                         'AvgA':'away_avg_win_odds2',
                                         'MaxH':'home_max_win_odds2',
                                         'MaxD':'draw_max_win_odds2',
                                         'MaxA':'away_max_win_odds2',
                                         'EloHome':'home_elo_rank',
                                         'EloAway':'away_elo_rank',
                                         })
    full_df['home_team_name'] = full_df['home_team']
    full_df['away_team_name'] = full_df['away_team']
    full_df['result'] = np.where(full_df['result']=='H',0,np.where(full_df['result']=='A',2,1))

    # get csv for team stadium locations
    teams_df = pd.read_csv(os.path.join(data_folder,'stadium_locations_final.csv'))
    teams_df = teams_df.rename(columns={'team':'team_name', 'geometry':'stadium_location'})
    teams_df['stadium_location'] = 'SRID=4326;'+ teams_df['stadium_location']

    eng = "postgresql+psycopg2://%s:%s@localhost:5432/soccer"%(user,db_pwd)
    engine = create_engine(eng)
    conn = engine.connect()

    sql = """
            DELETE from games;
            DELETE from teams;
            DELETE from leagues;
            DELETE from model_inputs;
            DELETE from predictions;
            DELETE from seasons;

          """
    conn.execute(sql)

    # add seasons
    sql = """
            INSERT INTO seasons (season)
                     VALUES ( %s );
         """
    for season in full_df['season'].unique():
        conn.execute(sql, (season,))

    # add leagues
    sql = """
            INSERT INTO leagues (league_name)
                     VALUES ( %s );
         """
    for l in full_df['league'].unique():
        conn.execute(sql, (l,))

    # add teams
    teams_df2 = teams_df.loc[:,['team_name','league', 'stadium_location']]
    db_utils.to_pg(df=teams_df2,table_name='teams', con=engine)

    sql = """
            SELECT *
            FROM teams
    """
    teams_df2 = pd.read_sql(sql=sql, con=engine)
    home_tm = teams_df2.rename(columns={'team_name':'home_team_name',
                                        'team_id':'home_team_id'})
    full_df = pd.merge(full_df, home_tm.loc[:,['home_team_name',
                                               'home_team_id']],
                       on='home_team_name', how='left')
    away_tm = teams_df2.rename(columns={'team_name':'away_team_name',
                                        'team_id':'away_team_id'})
    full_df = pd.merge(full_df, away_tm.loc[:,['away_team_name',
                                               'away_team_id']],
                       on='away_team_name',how='left')
    # season + game counter
    h = full_df.loc[:,['game_date', 'home_team_id', 'season']]
    aw = full_df.loc[:,['game_date', 'away_team_id', 'season']]
    h = h.rename(columns={'home_team_id':'team_id'})
    aw = aw.rename(columns={'away_team_id':'team_id'})
    r = pd.concat([h,aw],axis=0)
    r = r[~r['game_date'].isnull()]
    r = r.sort_values('game_date')
    r['game_count'] = r.groupby(['team_id','season']).cumcount()
    r['home_team_id'] = r['team_id']
    r['away_team_id'] = r['team_id']
    r['games_played_home'] = r['game_count']
    r['games_played_away'] = r['game_count']
    full_df = pd.merge(full_df,r.loc[:,['home_team_id', 'game_date','games_played_home']],
                       on=['home_team_id','game_date'], how='left')

    full_df = pd.merge(full_df,r.loc[:,['away_team_id', 'game_date','games_played_away']],
                       on=['away_team_id','game_date'], how='left')

    write_df = full_df.loc[:, ['game_date',
                    'league',
                    'season',
                    'home_team_id',
                    'away_team_id',
                    'home_team_name',
                    'away_team_name',
                    'games_played_home',
                    'games_played_away',
                    'home_goals_scored',
                    'away_goals_scored',
                    'result',
                    'home_shots',
                    'away_shots',
                    'home_shots_target',
                    'away_shots_target',
                    'home_fouls_committed',
                    'away_fouls_committed',
                    'home_corners_won',
                    'away_corners_won',
                    'home_yellows_conceded',
                    'away_yellows_conceded',
                    'home_reds_conceded',
                    'away_reds_conceded',
                    'home_avg_win_odds',
                    'draw_avg_win_odds',
                    'away_avg_win_odds',
                    'home_max_win_odds',
                    'draw_max_win_odds',
                    'away_max_win_odds',
                    'home_avg_win_odds2',
                    'draw_avg_win_odds2',
                    'away_avg_win_odds2',
                    'home_max_win_odds2',
                    'draw_max_win_odds2',
                    'away_max_win_odds2',
                    'home_elo_rank',
                    'away_elo_rank',]]
    write_df = write_df.fillna(value=0)
    for col in ['home_team_id',
                'away_team_id',
                'games_played_home',
                'games_played_away',
                'home_goals_scored',
                'away_goals_scored',
                'result',
                'home_shots',
                'away_shots',
                'home_shots_target',
                'away_shots_target',
                'home_fouls_committed',
                'away_fouls_committed',
                'home_corners_won',
                'away_corners_won',
                'home_yellows_conceded',
                'away_yellows_conceded',
                'home_reds_conceded',
                'away_reds_conceded']:
        write_df[col] = write_df[col].astype('int')
    for col in ['home_avg_win_odds',
                'draw_avg_win_odds',
                'away_avg_win_odds',
                'home_max_win_odds',
                'draw_max_win_odds',
                'away_max_win_odds',
                'home_avg_win_odds2',
                'draw_avg_win_odds2',
                'away_avg_win_odds2',
                'home_max_win_odds2',
                'draw_max_win_odds2',
                'away_max_win_odds2',
                'home_elo_rank',
                'away_elo_rank',]:
        write_df[col] = write_df[col].astype('float')
    write_df.to_csv('/Users/Steven/historical_soccer_data/to_db_debug.csv')
    db_utils.to_pg(df=write_df,table_name='games', con=engine)
    conn.close()


def _diff_stadium(data_folder):
    teams_df = pd.read_csv(os.path.join(data_folder,'stadium_locations_final.csv'))
    leagues = LEAGUES
    full_dfs = []
    for root, dirs, files in os.walk(data_folder):
        for fn in files:
            if fn.split('.')[0] in leagues and fn.endswith('.csv'):
                df_ = pd.read_csv(os.path.join(root, fn))
                df_['league'] = fn.split('.')[0]
                full_dfs.append(df_)
    full_df = pd.concat(full_dfs)
    full_df['Date'] = pd.to_datetime(full_df['Date'] )
    full_df.index = full_df['Date']
    full_df = full_df[~full_df['HomeTeam'].isnull()]

    t = full_df['HomeTeam'].unique()
    s = teams_df['team'].unique()
    d = np.setdiff1d(t,s)
    for i in d:
        print full_df['league'][full_df['HomeTeam']==i].iloc[0], i


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='hacky script to get all historical \
        data into db in one-time mode')
    parser.add_argument('-f', '--folder', type=str, required=True,
                        help='path to write data to before it goes off to db \
                        land and where stadium_locations_final.csv lives')
    parser.add_argument('-p', '--db_pwd', type=str, required=True,
                        help='password for db')
    parser.add_argument('-u', '--db_user', type=str, required=True,
                        help='user for db')
    parser.add_argument('-s', '--year_start_min', type=int, required=True,
                        help="the first season's start year")
    parser.add_argument('-e', '--year_start_max', type=int, required=True,
                        help="the last season's start year")
    args = parser.parse_args()
    agg_raw_data(data_folder=args.folder,
                 year_start_min=args.year_start_min,
                 year_start_max=args.year_start_max,
                 leagues=LEAGUES)
    elo_loop(leagues=LEAGUES,
             data_folder=args.folder)
    db_utils.drop_all_tbls(args.db_user, args.db_pwd)
    db_utils.create_db_tbls(args.db_user, args.db_pwd)
    populate_historical_data(data_folder=args.folder,
                             db_pwd=args.db_pwd,
                             user=args.db_user)
