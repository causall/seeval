So we're going to get the movies inidvidually then we'll average the rating scores and then
feed the algo the genres year and title, then we'll leverage whatever knowledge it's slurped
up to nudge it towards evaluating the movies such that the score hits the average solely on the data

The problem is the way I compute preferences is based on an aggregate statistic that would leak the test set of preferences,
so I need to compute preferences for only the selected set of movies

and then seapartely for the test set

This woudl rely on movie preference staying somewhat stationary

it's very possible that movies in either cohort don't have the same preferences
