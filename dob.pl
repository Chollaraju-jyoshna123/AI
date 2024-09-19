% Database of names and dates of birth
% Format: person(Name, date(Day, Month, Year)).

person('Alice', date(15, 5, 1990)).
person('Bob', date(22, 7, 1985)).
person('Charlie', date(10, 10, 2000)).
person('Diana', date(3, 3, 1995)).
person('Eve', date(1, 1, 1988)).

% Rule to find the date of birth by name
dob(Name, Date) :-
    person(Name, Date).

% Rule to find people born in a specific year
born_in_year(Year, Name) :-
    person(Name, date(_, _, Year)).

% Rule to find people born in a specific month
born_in_month(Month, Name) :-
    person(Name, date(_, Month, _)).

% Rule to find people born on a specific day
born_on_day(Day, Name) :-
    person(Name, date(Day, _, _)).