from sqlalchemy import create_engine
import os
import pandas as pd
import psycopg2
import requests
from StringIO import StringIO
import cStringIO
import datetime as dt
import re

"""
notes for mac
curl -o cloud_sql_proxy https://dl.google.com/cloudsql/cloud_sql_proxy.darwin.amd64
chmod +x cloud_sql_proxy
./cloud_sql_proxy -instances=soccer-262626:us-west1:soccer1=tcp:5432 -credential_file=/Users/steven/Documents/soccer-1da6bb794a28.json

"""

def cleanColumns(columns):
    cols = []
    for col in columns:
        col = col.replace(' ', '_')
        cols.append(col)
    return cols

def to_pg(df, table_name, con):
    output = cStringIO.StringIO()
    # ignore the index
    df.to_csv(output, sep='\t', header=False, index=False)
    output.getvalue()
    output.seek(0)
    connection = con.raw_connection()
    cursor = connection.cursor()
    cursor.copy_from(output, table_name, null="", columns=(df.columns))
    connection.commit()
    cursor.close()

def create_db_tbls(user,db_pwd):
    eng = "postgresql+psycopg2://%s:%s@localhost:5432/soccer"%(user,db_pwd)
    engine = create_engine(eng)
    conn = engine.connect()

    teams_sql = """
    CREATE TABLE seasons (
                        season varchar(20),
                        PRIMARY KEY(season)
                        );

    CREATE TABLE weather_stations (
                        station_id serial,
                        USAF varchar(20),
                        WBAN varchar(20),
                        station_location geometry(Point,4326),
                        UNIQUE(station_id),
                        PRIMARY KEY(station_id)
                        );

    CREATE TABLE weather_data (
                        station_id serial references weather_stations(station_id),
                        conditions varchar(50),
                        date date,
                        precip_time real,
                        precip_vol real,
                        pressure real,
                        temp real,
                        temp_qual real,
                        wind real,
                        wind_qual real
                        );

    CREATE TABLE leagues (
                        league_name varchar(20),
                        PRIMARY KEY(league_name)
                        );

    CREATE TABLE teams (
                        team_id serial,
                        team_name varchar(30),
                        league varchar(20) references leagues(league_name),
                        stadium_location geometry(Point,4326),
                        UNIQUE(team_name),
                        PRIMARY KEY(team_id)
                        );

    CREATE TABLE games (
                        game_id serial,
                        game_date date,
                        home_team_id serial NOT NULL references teams(team_id),
                        away_team_id serial NOT NULL references teams(team_id),
                        home_team_name varchar(30) NOT NULL references teams(team_name),
                        away_team_name varchar(30) NOT NULL references teams(team_name),
                        league varchar(20) references leagues(league_name),
                        season varchar(20) references seasons(season),
                        games_played_home int,
                        games_played_away int,
                        home_goals_scored int,
    					away_goals_scored int,
    					home_shots int,
    					away_shots int,
                        home_shots_target int,
                        away_shots_target int,
    					home_fouls_committed int,
    					away_fouls_committed int,
    					home_corners_won int,
    					away_corners_won int,
    					home_yellows_conceded int,
    					away_yellows_conceded int,
    					home_reds_conceded int,
    					away_reds_conceded int,
    					home_elo_rank double precision,
    					away_elo_rank double precision,
                        home_formation varchar(10),
                        away_formation varchar(10),
                        home_avg_win_odds real,
                        away_avg_win_odds real,
                        draw_avg_win_odds real,
    					result int,
                        UNIQUE(games_played_home, home_team_id, season),
                        UNIQUE(games_played_away, away_team_id, season),
                        PRIMARY KEY(game_id)
                        );

    CREATE TABLE model_inputs (
                                game_id serial NOT NULL references games(game_id),
                                home_team_name varchar(30) NOT NULL references teams(team_name),
                                away_team_name varchar(30) NOT NULL references teams(team_name),
                                home_avg_win_odds numeric,
                                away_avg_win_odds numeric,
                                draw_avg_win_odds numeric,
                                away_points_to_date numeric,
                                away_pct_points_won_to_date numeric,
                                away_pct_points_won_ma1 numeric,
                                away_pct_points_won_ma3 numeric,
                                away_pct_points_won_ma5 numeric,
                                away_goals_scored_mean_to_date numeric,
                                away_goals_scored_ma1 numeric,
                                away_goals_scored_ma3 numeric,
                                away_goals_scored_ma5 numeric,
                                away_goals_conceded_mean_to_date numeric,
                                away_goals_conceded_ma1 numeric,
                                away_goals_conceded_ma3 numeric,
                                away_goals_conceded_ma5 numeric,
                                away_shots_taken_mean_to_date numeric,
                                away_shots_taken_ma1 numeric,
                                away_shots_taken_ma3 numeric,
                                away_shots_taken_ma5 numeric,
                                away_shots_taken_on_target_mean_to_date numeric,
                                away_shots_taken_on_target_ma1 numeric,
                                away_shots_taken_on_target_ma3 numeric,
                                away_shots_taken_on_target_ma5 numeric,
                                away_shots_allowed_mean_to_date numeric,
                                away_shots_allowed_ma1 numeric,
                                away_shots_allowed_ma3 numeric,
                                away_shots_allowed_ma5 numeric,
                                away_shots_allowed_on_target_mean_to_date numeric,
                                away_shots_allowed_on_target_ma1 numeric,
                                away_shots_allowed_on_target_ma3 numeric,
                                away_shots_allowed_on_target_ma5 numeric,
                                away_corners_won_mean_to_date numeric,
                                away_corners_won_ma1 numeric,
                                away_corners_won_ma3 numeric,
                                away_corners_won_ma5 numeric,
                                away_corners_conceded_mean_to_date numeric,
                                away_corners_conceded_ma1 numeric,
                                away_corners_conceded_ma3 numeric,
                                away_corners_conceded_ma5 numeric,
                                away_off_efficiency_mean_to_date numeric,
                                away_def_efficiency_mean_to_date numeric,
                                away_off_efficiency_ma1 numeric,
                                away_def_efficiency_ma1 numeric,
                                away_off_efficiency_ma3 numeric,
                                away_def_efficiency_ma3 numeric,
                                away_off_efficiency_ma5 numeric,
                                away_def_efficiency_ma5 numeric,
                                away_prior_elo_rank_ma1 numeric,
                                away_prior_elo_rank_ma3 numeric,
                                away_prior_elo_rank_ma5 numeric,
                                home_points_to_date numeric,
                                home_pct_points_won_to_date numeric,
                                home_pct_points_won_ma1 numeric,
                                home_pct_points_won_ma3 numeric,
                                home_pct_points_won_ma5 numeric,
                                home_goals_scored_mean_to_date numeric,
                                home_goals_scored_ma1 numeric,
                                home_goals_scored_ma3 numeric,
                                home_goals_scored_ma5 numeric,
                                home_goals_conceded_mean_to_date numeric,
                                home_goals_conceded_ma1 numeric,
                                home_goals_conceded_ma3 numeric,
                                home_goals_conceded_ma5 numeric,
                                home_shots_taken_mean_to_date numeric,
                                home_shots_taken_ma1 numeric,
                                home_shots_taken_ma3 numeric,
                                home_shots_taken_ma5 numeric,
                                home_shots_taken_on_target_mean_to_date numeric,
                                home_shots_taken_on_target_ma1 numeric,
                                home_shots_taken_on_target_ma3 numeric,
                                home_shots_taken_on_target_ma5 numeric,
                                home_shots_allowed_mean_to_date numeric,
                                home_shots_allowed_ma1 numeric,
                                home_shots_allowed_ma3 numeric,
                                home_shots_allowed_ma5 numeric,
                                home_shots_allowed_on_target_mean_to_date numeric,
                                home_shots_allowed_on_target_ma1 numeric,
                                home_shots_allowed_on_target_ma3 numeric,
                                home_shots_allowed_on_target_ma5 numeric,
                                home_corners_won_mean_to_date numeric,
                                home_corners_won_ma1 numeric,
                                home_corners_won_ma3 numeric,
                                home_corners_won_ma5 numeric,
                                home_corners_conceded_mean_to_date numeric,
                                home_corners_conceded_ma1 numeric,
                                home_corners_conceded_ma3 numeric,
                                home_corners_conceded_ma5 numeric,
                                home_off_efficiency_mean_to_date numeric,
                                home_def_efficiency_mean_to_date numeric,
                                home_off_efficiency_ma1 numeric,
                                home_def_efficiency_ma1 numeric,
                                home_off_efficiency_ma3 numeric,
                                home_def_efficiency_ma3 numeric,
                                home_off_efficiency_ma5 numeric,
                                home_def_efficiency_ma5 numeric,
                                home_prior_elo_rank_ma1 numeric,
                                home_prior_elo_rank_ma3 numeric,
                                home_prior_elo_rank_ma5 numeric,
                                UNIQUE(game_id)
                               );

    CREATE TABLE predictions (
                              game_id serial NOT NULL  references games(game_id),
                              prob_home double precision NOT NULL,
                              prob_draw double precision NOT NULL,
                              prob_away double precision NOT NULL,
                              UNIQUE(game_id)
                              );
    """
    conn.execute(teams_sql)
    conn.close()

def drop_all_tbls(user,db_pwd):
    eng = "postgresql+psycopg2://%s:%s@localhost:5432/soccer"%(user,db_pwd)
    engine = create_engine(eng)
    conn = engine.connect()
    sql="""
            DROP TABLE IF EXISTS games CASCADE;
            DROP TABLE IF EXISTS teams CASCADE;
            DROP TABLE IF EXISTS leagues CASCADE;
            DROP TABLE IF EXISTS model_inputs CASCADE;
            DROP TABLE IF EXISTS predictions CASCADE;
            DROP TABLE IF EXISTS seasons CASCADE;
            DROP TABLE IF EXISTS weather_stations CASCADE;
            DROP TABLE IF EXISTS weather_data CASCADE;

    """
    conn.execute(sql)
    conn.close()
