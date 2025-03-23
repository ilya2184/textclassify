import re
import os
import Levenshtein
from concurrent.futures import ProcessPoolExecutor
from rapidfuzz import fuzz, utils
from app_logging import writelog


def clean_text(text):
    text = utils.default_process(text)
    return text

def compare_single_contact(contact, checklist, threshold_high, threshold_low):
    contact_name = contact["name"]
    contact_name_cleaned = clean_text(contact_name)
    matches_high = []
    matches_low = []

    if contact_name_cleaned == '':
        return contact_name, matches_high, matches_low
    
    for item in checklist:
        checklist_name = item["name"]
        checklist_name_cleaned = clean_text(checklist_name)
        if checklist_name_cleaned == '':
            continue
        score = fuzz.token_sort_ratio(contact_name_cleaned, checklist_name_cleaned)
        if score >= threshold_high:
            matches_high.append((checklist_name, score))
        elif score >= threshold_low:
            matches_low.append((checklist_name, score))

    return contact_name, matches_high, matches_low

def compare_contacts(contacts, checklist, threshold_high=97, threshold_low=80):
    results = {}
    
    # Определяем количество потоков
    num_cores = os.cpu_count()
    num_threads = (num_cores) // 2
    if num_threads <= 0:
        num_threads = 1
    contacts_len = len(contacts)
    chunk_size = contacts_len // num_threads + (contacts_len % num_threads > 0)

    # Разделяем список contacts на части
    chunks = [contacts[i:i + chunk_size] for i in range(0, contacts_len, chunk_size)]
    checklist_len = len(checklist)
    writelog(f"List1 {contacts_len} and list2 {checklist_len} comparison started in {num_threads} threads.")

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for chunk in chunks:
            futures.append(executor.submit(process_chunk, chunk, checklist, threshold_high, threshold_low))

        for future in futures:
            contact_results = future.result()
            for contact_name, matches_high, matches_low in contact_results:
                # Добавляем контакт в результат только если есть совпадения в high или low
                if matches_high or matches_low:
                    results[contact_name] = {
                        "matches_high": matches_high,
                        "matches_low": matches_low,
                    }
    
    return results

def process_chunk(chunk, checklist, threshold_high, threshold_low):
    results = []
    for contact in chunk:
        contact_name, matches_high, matches_low = compare_single_contact(contact, checklist, threshold_high, threshold_low)
        results.append((contact_name, matches_high, matches_low))
    return results

def find_best_match(string_list, text, length_penalty_factor):
    cleaned_text = clean_text(text)
    best_match = None
    best_score = float('inf')

    for current_string in string_list:
        cleaned_current_string = clean_text(current_string)
        if len(cleaned_current_string) == 0:
            continue
        for i in range(len(cleaned_text) - len(cleaned_current_string) + 1):
            substring = cleaned_text[i:i+len(cleaned_current_string)]
            distance = Levenshtein.distance(cleaned_current_string, substring)
            # Нормализуем расстояние делением на длину строки и добавляем штраф за длину
            normalized_distance = (distance / len(cleaned_current_string)) + (length_penalty_factor * len(cleaned_current_string))
            
            # Используем нормализованное расстояние для определения лучшего совпадения
            if normalized_distance < best_score:
                best_score = normalized_distance
                best_match = current_string
    return best_match