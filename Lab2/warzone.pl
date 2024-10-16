% Факты с одним аргументом
%-------------------------
% типы оружия
weapon_type("assault rifle").   % штурмовая винтовка
weapon_type("combat rifle").    % боевая винтовка
weapon_type("submachine gun").  % пистолет-пулемет
weapon_type("shotgun").         % дробовик
weapon_type("machine Gun").     % ручной пулемет
weapon_type("infantry rifle").  % пехотная винтовка
weapon_type("sniper rifle").    % снайперская винтовка
weapon_type("pistol").          % пистолет
weapon_type("close combat").    % ближний бой

% штурмовые винтовки
weapon("M16").
weapon("MCW").
weapon("Caste 74u").

% боевые винтовки
weapon("MTZ-762").
weapon("BAS-B").
weapon("TAQ-V").

% пистолет-пулеметы
weapon("AMR9").
weapon("VEL 46").
weapon("PDSW 528").

% дробовики
weapon("Riveter").
weapon("Lockwood 680").
weapon("Haymaking").

% ручные пулеметы
weapon("PKK").
weapon("762 machine gun").
weapon("HCR 56").

% пехотные винтовки
weapon("Kar98k").
weapon("SP-R 208").
weapon("Lockwood-2").

% снайперские винтовки
weapon("MORS").
weapon("KATT-AMR").
weapon("Kappak .300").

% пистолеты
weapon("Reinetti").
weapon("P890").
weapon("GS Magna").

% ближний бой
weapon("Pickaxe").
weapon("Kerambit").
weapon("Paired sickles").

% калибр 
caliber("5.56mm").
caliber("7.62mm").
caliber("9mm").
caliber("12mm").
caliber("50mm").

% броня 
armor("reinforced pouch").             % Усиленный подсумок, вмещает 2 бронепластины
armor("medical body armor").           % Медицинский бронежилет, позволяет быстрее реанемировать напарников
armor("signalman's bulletproof vest"). % Бронежилет связиста, БПЛА проводит сканирование в два раза чаще
armor("stealth body armor").           % Стелс-бронежилет, игрок становится полностью невидим для БПЛА и МБПЛА

% зоны
zone("safe").                % безопасная зона
zone("gas").                 % зона с газом
zone("precision airstrike"). % зона авиационной атаки 
zone("killstreaks").         % опасные объекты

% карты
map("fortress of fortune").  % крепость фортуны
map("vernadsk").             % верданск



% Факты с двумя аргументами
%-------------------------
% восстановление здоровья
medkit("stim", 100).           % стимулятор, восполняет полное здоровье
medkit("self-revive kit", 10). % самореаниматор, позволяет подняться после того как сбили

% типы оружия и их классификация
weapon_type_class("M16", "assault rifle").
weapon_type_class("MCW", "assault rifle").
weapon_type_class("Caste 74u", "assault rifle").

weapon_type_class("MTZ-762", "combat rifle").
weapon_type_class("BAS-B", "combat rifle").
weapon_type_class("TAQ-V", "combat rifle").

weapon_type_class("AMR9", "submachine gun").
weapon_type_class("VEL 46", "submachine gun").
weapon_type_class("PDSW 528", "submachine gun").

weapon_type_class("Riveter", "shotgun").
weapon_type_class("Lockwood 680", "shotgun").
weapon_type_class("Haymaking", "shotgun").

weapon_type_class("PKK", "machine Gun").
weapon_type_class("762 machine gun", "machine Gun").
weapon_type_class("HCR 56", "machine Gun").

weapon_type_class("Kar98k", "infantry rifle").
weapon_type_class("SP-R 208", "infantry rifle").
weapon_type_class("Lockwood-2", "infantry rifle").

weapon_type_class("MORS", "sniper rifle").
weapon_type_class("KATT-AMR", "sniper rifle").
weapon_type_class("Kappak .300", "sniper rifle").

weapon_type_class("Reinetti", "pistol").
weapon_type_class("P890", "pistol").
weapon_type_class("GS Magna", "pistol").

weapon_type_class("Pickaxe", "close combat").
weapon_type_class("Kerambit", "close combat").
weapon_type_class("Paired sickles", "close combat").


% тип боеприпасов у оружия
% 5.56mm
ammunition_calibre("M16", "5.56mm").
ammunition_calibre("MCW", "5.56mm").
ammunition_calibre("Caste 74u", "5.56mm").
% 7.62mm
ammunition_calibre("MTZ-762", "7.62mm").
ammunition_calibre("BAS-B", "7.62mm").
ammunition_calibre("TAQ-V", "7.62mm").
ammunition_calibre("PKK", "7.62mm").
ammunition_calibre("762 machine gun", "7.62mm").
ammunition_calibre("HCR 56", "7.62mm").
ammunition_calibre("Kar98k", "7.62mm").
ammunition_calibre("SP-R 208", "7.62mm").
ammunition_calibre("Lockwood-2", "7.62mm").
% 9mm
ammunition_calibre("AMR9", "9mm").
ammunition_calibre("VEL 46", "9mm").
ammunition_calibre("PDSW 528", "9mm").
ammunition_calibre("Reinetti", "9mm").
ammunition_calibre("P890", "9mm").
ammunition_calibre("GS Magna", "9mm").
% 12mm
ammunition_calibre("Riveter", "12mm").
ammunition_calibre("Lockwood 680", "12mm").
ammunition_calibre("Haymaking", "12mm").
% 50mm
ammunition_calibre("MORS", "50mm").
ammunition_calibre("KATT-AMR", "50mm").
ammunition_calibre("Kappak .300", "50mm").
% -mm, ближний бой
ammunition_calibre("Pickaxe", "-").
ammunition_calibre("Kerambit", "-").
ammunition_calibre("Paired sickles", "-").


% хп игроков
player_health("Maria", 100).
player_health("Djeno", 80).
player_health("Egor", 25).
player_health("Timur", 1).
player_health("Alex", 75).

% расположение игроков относительно зон
player_location("Maria", "safe").
player_location("Djeno", "gas").
player_location("Egor", "safe").
player_location("Timur", "precision airstrike").
player_location("Alex", "killstreaks").

% броня игроков
player_armor("Maria", 150).
player_armor("Djeno", 50).
player_armor("Egor", 50).
player_armor("Timur", 0).
player_armor("Alex", 100).


% Правила
%-------------------------
% Оружие подходит для дальнего боя, если оно является пехотным или снайперским, Caste 74u или использует 7.62.
long_range_weapon(Gun) :-
weapon_type_class(Gun, "infantry rifle");
weapon_type_class(Gun, "sniper rifle");
Gun = "Caste 74u";
ammunition_calibre(Gun, "7.62mm").

% Оружие подходит для ближнего боя, если это M16, дробовик или оно использует патроны калибра 9 мм.
low_range_weapon(Weapon) :-
    weapon_type_class(Weapon, "M16");
    weapon_type_class(Weapon, "shotgun");
    ammunition_calibre(Weapon, "9mm").

% Оружие подходит для средних дистанций, если это штурмовая или боевая винтовка, или если оно использует патроны калибра 5.56 мм.
middle_range_weapon(Weapon) :-
    weapon_type_class(Weapon, "assault rifle");
    weapon_type_class(Weapon, "combat rifle");
    ammunition_calibre(Weapon, "5.56mm").

% Использовать стимулятор, если здоровье игрока находится между 5 и 90%
use_medkit(Player, "stim") :-
    player_health(Player, Health),
    Health > 5,
    Health =< 90.

% Использовать самореаниматор, если здоровье игрока ниже 5%
use_medkit(Player, "self-revive kit") :-
    player_health(Player, Health),
    Health =< 5.

% Проверка угрожает ли игроку смерть от какой-нибудь зоны
player_in_danger(Player) :-
    \+ player_location(Player, "safe"),
    player_health(Player, Health),
    Health < 50.
%-------------------------


% запросы
%-------------------------
%middle_range_weapon(Gun).
%low_range_weapon(Gun).
%long_range_weapon(Gun). Какие оружия подходят для дальнего боя
%ammunition_calibre("M16", "7.62mm"). Использует ли мини калибр 7.62
%use_medkit(Player, "stim"). Какие игроки должны использовать первую аптечку
%weapon_type_class(Gun, "sniper rifle"). Какое оружие относится к снайперкам
%player_location(Player, "safe"). Какие игроки находятся в безопасной зоне
%use_medkit("Djeno", Medkit). Какое средство лечения должен использовать Djeno
%player_health("Timur", Health), use_medkit("Timur", Medkit). Какую аптеку должен использовать Timur
%long_range_weapon(Gun), ammunition_calibre(Gun, "7.62mm"). Какое оружие использует патроны 7.62 мм и подходит для дальнего боя?
%player_location(Player, "gas"); player_location(Player, "precision airstrike"). Какие игроки находятся в зоне с газом или в зоне авиационной атаки
%-------------------------