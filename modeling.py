import pandas as pd
import numpy as np
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import pickle
import db_utils
from sqlalchemy.engine import create_engine
import prettytable
from StringIO import StringIO
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

ALL_LEAGUES = ['Premier', 'Championship', 'SerieA', 'laLiga1', 'Bundesliga1', 'Championnat']
SEASON_SPLITS = {'train_test':10,'validate':6}


def pprint_df(df):
    output = StringIO()
    df.to_csv(output)
    output.seek(0)
    pt = prettytable.from_csv(output)
    print pt


class TrainTestValidate:
    def __init__(self,user,db_pwd,train_stop_date,drop_na=False):
        self.drop_na = drop_na
        self.train_stop_date = train_stop_date
        eng = "postgresql+psycopg2://%s:%s@localhost:5432/soccer"%(user,db_pwd)
        engine = create_engine(eng)
        conn = engine.connect()
        self._engine = engine
        self._conn = conn
        self.run()

    def _get_seasons(self):
        SQL = """
        SELECT * FROM seasons
        """
        seasons = pd.read_sql(sql=SQL, con=self._engine)
        order = seasons['season'].str[0:4].sort_values().index
        seasons = seasons.reindex(order)
        train_test_seasons = seasons[0:SEASON_SPLITS['train_test']]
        validate_seasons = seasons[SEASON_SPLITS['train_test']:SEASON_SPLITS['train_test']+SEASON_SPLITS['validate']]
        return train_test_seasons, validate_seasons

    def _get_model_inputs(self):
        SQL = """
        SELECT * FROM model_inputs

        """
        model_input_df = pd.read_sql(sql=SQL, con=self._engine)
        model_input_df.index = model_input_df['game_id']
        return model_input_df

    def _get_games(self):
        SQL = """
        SELECT * FROM games
        """
        games_df = pd.read_sql(sql=SQL, con=self._engine)
        return games_df

    def _create_train_test(self):
        train_test_seasons, validate_seasons = self._get_seasons()
        full_games_df = self._get_games()
        model_input_df = self._get_model_inputs()
        games_df = full_games_df.loc[:,['game_id','league','season','result','game_date']]
        full_df = pd.merge(games_df, model_input_df,on=['game_id'])
        full_df.index = full_df['game_id']
        full_df = full_df[full_df['season'].isin(train_test_seasons['season'])]
        full_df['game_date'] = pd.to_datetime(full_df['game_date'])
        full_df.index = full_df['game_date']
        if self.drop_na is True:
            full_df = full_df.dropna()

        train_test_data = {}
        for l in games_df['league'].unique():
            df = full_df[full_df['league']==l]
            train = df[:self.train_stop_date]
            test = df[self.train_stop_date:]

            features = list(model_input_df.columns)
            features = [i for i in features if i not in ['home_team_name','away_team_name']]

            X_train_df = train.loc[:,features]
            y_train = train['result']

            X_test_df = test.loc[:,features]
            y_test = test['result']

            train_test_data[l] = {'train_df':train,
                                  'test_df':test,
                                  'X_train_df':X_train_df,
                                  'y_train':y_train,
                                  'X_test_df':X_test_df,
                                  'y_test':y_test,
                                  'games_df':full_games_df[full_games_df['league']==l]
                                 }
        self.train_test_data = train_test_data

    def run(self):
        self._create_train_test()

class XgbFlow:
    def __init__(self,TrainTestValidate,league,print_output):
        self.tt = TrainTestValidate
        self.league = league
        self._print_output = print_output
        self.initial_model = None
        self._run_intialize()

    def _convert_to_binary_class(self):
         for ix,data in self.tt.train_test_data.iteritems():
                data['y_train'] = np.where(data['y_train']==0,1,0)
                data['y_test'] = np.where(data['y_test']==0,1,0)


    def _gridsearch(self):
        for ix,data in self.tt.train_test_data.iteritems():
            if ix==self.league:
                X_train = data['X_train_df'].drop('game_id',axis=1)
                y_train = data['y_train']
                params = {'max_depth': [3,5,7],
                          'min_child_weight': [3,5,7],
                          'learning_rate': [0.01,0.001],
                          'colsample_bytree': [0.5,0.7],
                          'subsample': [0.4,0.6,0.8],
                          'max_delta_step':[0,1]
                         }
                ind_params = {'n_estimators': 4000,
                              'seed':26,
                              'early_stopping_rounds':200,
                              'objective':'binary:logistic'
                             }
                xgb_model = GridSearchCV(xgb.XGBClassifier(ind_params),
                                         params,
                                         scoring='neg_log_loss',
                                         cv=3,
                                         n_jobs=-1)
                initial_xgb_model = xgb_model.fit(X_train, y_train)
                y = initial_xgb_model.best_estimator_.feature_importances_
                x = X_train.columns
                feat_imp_df = pd.DataFrame({'features':x,'values':y})
                feat_imp_df = feat_imp_df.sort_values('values')
                self.initial_model = {'best_insample_params':initial_xgb_model.best_params_,
                                      'best_insample_score':initial_xgb_model.best_score_,
                                      'insample_feat_imp_df':feat_imp_df,
                                      'insample_result_df':np.nan}


    def train_n_predict(self,params,train_df,y_train,test_df,y_test):
        xgb_matrix_train = xgb.DMatrix(train_df,
                                       label=y_train,
                                       missing = np.nan
                                      )
        xgb_matrix_test = xgb.DMatrix(test_df,
                                      label=y_test,
                                      missing = np.nan
                                     )
        final_params = {'eta':params['learning_rate'],
                        'seed':26,
                        "silent":1,
                        'subsample': params['subsample'],
                        'colsample_bytree': params['colsample_bytree'],
                        'objective': 'binary:logistic',
                        'max_depth':params['max_depth'],
                        'min_child_weight':params['min_child_weight'],
                        'eval_metric':'logloss'}

        trained_xgb_model = xgb.train(params=final_params,
                                      verbose_eval=False,
                                      dtrain=xgb_matrix_train,
                                      num_boost_round=6000,
                                      early_stopping_rounds=800,
                                      evals=[(xgb_matrix_train,'train'),
                                              (xgb_matrix_test,'eval')]
                                      )
        y_scores = trained_xgb_model.predict(xgb_matrix_test)
        return y_scores

    def run_model(self,max_date):
        data = self.tt.train_test_data[self.league]
        games_df = self.tt.train_test_data[self.league]['games_df']
        params = self.initial_model['best_insample_params']
        train_df = data['X_train_df']
        y_train = data['y_train']
        test_df = data['X_test_df']
        y_test = data['y_test']
        y_test_df = pd.DataFrame({'y_test':y_test,'game_date':test_df.index})
        y_train_df = pd.DataFrame({'y_train':y_train,'game_date':train_df.index})
        y_train_df.index = y_train_df['game_date']
        y_test_df.index = y_test_df['game_date']
        dates = test_df[:max_date].index.unique()
        game_ids = []
        y_scores = []
        for d in dates:
            game_ids_ = test_df[d:d]['game_id']
            train_df_ = train_df[:d].drop('game_id',axis=1)
            test_df_ = test_df[d:d].drop('game_id',axis=1)
            y_train_ = y_train_df[:d]
            y_test_ = y_test_df[d:d]
            y_scores_ = self.train_n_predict(params=params,
                                             y_train=y_train_['y_train'],
                                             y_test=y_test_['y_test'],
                                             train_df=train_df_,
                                             test_df=test_df_)
            y_scores.extend(y_scores_)
            game_ids.extend(game_ids_)
            out = games_df[games_df['game_id'].isin(game_ids_)].loc[:,['home_team_name',
                                                                       'away_team_name',
                                                                       'home_goals_scored',
                                                                       'away_goals_scored',
                                                                       'home_avg_win_odds',
                                                                       'draw_avg_win_odds']
                                                                   ]
            y_scores_round = [ round(elem, 2) for elem in y_scores_]
            out.index = out['home_team_name']
            out['score'] = out['home_goals_scored'].astype(str) + ' | ' + out['away_goals_scored'].astype(str)
            out['y_scores'] = y_scores_round
            out = out.drop(['home_team_name','home_goals_scored','away_goals_scored'],axis=1)
            if self._print_output:
                print d.strftime('%Y-%m-%d')
                print ''
                pprint_df(out)
                print ''
                print ''
                print ''
                print '------------------'
        result_df = pd.DataFrame({'game_id':game_ids,
                                  'y_score':y_scores})
        result_df = pd.merge(result_df,games_df.loc[:,['game_id','result','home_avg_win_odds','game_date']],
                             on='game_id')
        result_df['home_win'] = np.where(result_df['result']==0,1,0)
        self.result_df = result_df

    def _run_intialize(self):
        print 'finding initial model paramaters'
        self._convert_to_binary_class()
        self._gridsearch()
