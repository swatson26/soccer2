

## Set up db

### install brew
`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

### install via brew
`brew update`
`brew doctor`
`brew install postgresql`

### start, and create db
`brew services start postgresql` </br>
`psql postgres`
* then you can exit psql with `\q`
* log in to rds with postico if ya want and run `create database soccer`

### add gis support
in postico run:
```
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_topology;
```

### run
`python historical_ingestion.py --help`
and put in some info
use `-s 6 -e 19` for season start and end`
this code is pretty janky, sorry.

### create model features
`python feature_creation.py -u XXXX -p XXXX`
* if you want to create new features, add to this script
