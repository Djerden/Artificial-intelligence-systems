import sys
import os
from swiplserver import PrologMQI

def get_knowledge_base_path():
    base_path = os.path.join(os.path.dirname(__file__), "warzone.pl")
    return os.path.normpath(base_path).replace("\\", "/") 

KNOWLEDGE_BASE = get_knowledge_base_path()

REQUESTS = {
    "range_weapon": "найди оружие для",
    "calibre_weapon": "найди оружие с калибром",
    "use_medkit": "какую аптечку должен использовать",
    "weapon_class": "какое оружие относится к классу",
    "weapon_type_list": "какие типы оружия есть"
}

def get_user_request():
    while True:
        display_requests()
        user_input = input("Ваш запрос: ").strip()

        if user_input.lower() == "/q":
            print("Работа завершена.")
            sys.exit(0)

        request_type, parameter = parse_user_input(user_input)
        if request_type:
            return request_type, parameter
        print("Неизвестный тип запроса, попробуйте снова.")

def display_requests():
    print("\n")
    print("Что Вы хотите узнать?")
    print(f"1 {REQUESTS['range_weapon']} <тип боя> (дальний, ближний, средний)")
    print(f"2 {REQUESTS['calibre_weapon']} <калибр mm> (5.56mm, 7.62mm, 9mm, 12mm, 50mm)")
    print(f"3 {REQUESTS['use_medkit']} <имя игрока> (Maria, Djeno, Egor, Timur, Alex)")
    print(f"4 {REQUESTS['weapon_class']} <тип оружия>")
    print(f"5 {REQUESTS['weapon_type_list']}")
    print("\n/q - для выхода из программы")
    print("\n")

def parse_user_input(user_input):
    for request_type, request_phrase in REQUESTS.items():
        if request_phrase in user_input:
            parameter = user_input.replace(request_phrase, "").strip()
            return request_type, parameter
    return None, None

def form_prolog_query(request_type, parameter):
    if request_type == "range_weapon":
        if parameter == "дальний":
            return "findall(Gun, long_range_weapon(Gun), Guns)."
        elif parameter == "ближний":
            return "findall(Gun, low_range_weapon(Gun), Guns)."
        elif parameter == "средний":
            return "findall(Gun, middle_range_weapon(Gun), Guns)."
        else:
            return None 
        
    elif request_type == "use_medkit":
        return f"use_medkit({parameter}, Medkit)."
    
    elif request_type == "calibre_weapon":
        return f"findall(Gun, ammunition_calibre(Gun, \"{parameter}\"), Guns)."
    
    elif request_type == "weapon_class":
        return f"findall(Weapon, weapon_type_class(Weapon, \"{parameter}\"), Weapons)."
    
    elif request_type == "weapon_type_list":
        return f"findall(Weapon, weapon_type(Weapon), Weapons)."
    

def process_request(request_type, parameter): 
    prolog_query = form_prolog_query(request_type, parameter)

    if prolog_query is None:
        print("Ошибка: Неправильный параметр запроса.")
        return

    with PrologMQI() as mqi:
        with mqi.create_thread() as prolog_thread:
            prolog_thread.query(f"consult('{KNOWLEDGE_BASE}').")
            response = prolog_thread.query(prolog_query)

            if response:
                handle_response(response, request_type, parameter)
            else:
                print("Запрос не дал результатов.")

def handle_response(response, request_type, parameter):
    if request_type == "range_weapon":
        guns = response[0]["Guns"]
        print(f"Оружие для {parameter} боя: {', '.join(guns) if guns else 'не найдено.'}")
    elif request_type == "calibre_weapon":
        guns = response[0]["Guns"]
        print(f"Оружие с калибром {parameter}: {', '.join(guns) if guns else 'не найдено.'}")
    elif request_type == "weapon_class":
        weapons = response[0]["Weapons"]
        print(f"Оружие типа '{parameter}': {', '.join(weapons) if weapons else 'не найдено.'}")
    elif request_type == "use_medkit":
        medkit = response[0]["Medkit"]
        print(f"{parameter} должен использовать: {medkit}.")
    elif request_type == "weapon_type_list":
        weapons = response[0]["Weapons"]
        print(f"Типы оружия: {', '.join(weapons) if weapons else 'не найдено.'}")


while True:
    request_type, parameter = get_user_request()
    process_request(request_type, parameter)
