# Working with the new data
#import mysklearn.mypytable
from mysklearn.myclassifiers import MyRandomForestsClassifier
from mysklearn.mypytable import MyPyTable 
from mysklearn import myutils

nba_data = MyPyTable().load_from_file("data/input/season_data.txt")
nba_stats = MyPyTable().load_from_file("data/input/season_stats.txt")
nba_data.drop_column(['Team','W','L','W/L%','Finish','SRS','','Pace','Rel Pace','Rel ORtg','Rel DRtg','Coaches','Top WS'])
nba_stats.drop_column([''])
nba_data.add_keyid_column()
nba_stats.add_keyid_column()
nba_all_data = nba_data.perform_full_outer_join(nba_stats, ["id", "Season", "Lg"])
myutils.label_playoff(nba_all_data)
nba_all_data.remove_rows_with_missing_values()

nba_all_data.save_to_file("data/output/trial1.txt")