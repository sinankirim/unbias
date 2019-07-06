import spacy
import numpy as np
import pandas as pd
from spacy.pipeline import Sentencizer
import json

def prepare_model(model="en_core_web_md"):
    nlp = spacy.load(model)
    sentencizer = Sentencizer()
    nlp.add_pipe(sentencizer, before="parser")
    return nlp

def prepare_namedata(filename="GenderCheckerNames.csv"):
    gencheck = pd.read_csv(filename, encoding="latin-1")
    gencheck_ = []
    for item in list(gencheck.values):
        gencheck_.append((item[0].replace("\"","")).split(","))
    for entry in gencheck_:
        entry[0] = entry[0].lower()
    return np.array(gencheck_)

def prepare_resolves(filename="persons_gender.xlsx"):
    pergen = pd.read_excel(filename)
    return pergen

def prepare_json(jsonname="genderedwords.json"):
    data = None
    with open(jsonname, "r") as f:
        data = json.load(f)
    return data

def documentize(text, nlp):
    doc = nlp(text)
    return doc

def flag_dependency(sentence, word_data):
    score = 0
    root = None
    object = None
    subject = None
    possess = []
    objpron = []
    for tok in sentence:
        if tok.dep_ == "ROOT":
            root = tok
            break
    search = list(root.children)
    while search != [] and object is None:
        for tok in search:
            if tok.dep_ in ["dobj", "iobj", "pobj", "attr"]:
                object = tok
                break
        temp = []
        for tok in search:
            temp.extend(list(tok.children))
        search = [i for i in temp]
    search = list(root.children)
    while search != [] and subject is None:
        for tok in search:
            if tok.dep_ in ["csubj", "csubjpass", "nsubj", "nsubjpass", "xsubj"]:
                subject = tok
                break
        temp = []
        for tok in search:
            temp.extend(list(tok.children))
        search = [i for i in temp]
    search = list(object.children) if object is not None else []
    while search != [] and possess == []:
        for tok in search:
            if tok.dep_ == "poss":
                possess.append(tok)
        temp = []
        for tok in search:
            temp.extend(list(tok.children))
        search = [i for i in temp]
    search = list(object.children) if object is not None else []
    while search != [] and objpron == []:
        for tok in search:
            if tok.pos_ == "PRON" and tok.dep_ == "pobj":
                objpron.append(tok)
        temp = []
        for tok in search:
            temp.extend(list(tok.children))
        search = [i for i in temp]
    print(root, object, subject, possess, objpron)
    for candidate in possess:
        #if possess.text == word_data["poss_adjs"]["male"] and object.text not in word_data["pronouns"]["object"]["male"]:
        #    score = score + (-1)
        #if possess.text == word_data["poss_adjs"]["female"] and object.text not in word_data["pronouns"]["object"]["female"]:
        #    score = score + 1
        if candidate.text == word_data["poss_adjs"]["male"] and subject.text not in word_data["pronouns"]["subject"]["male"] and subject.text not in word_data["biased_words"]["other_gendered"]["male"]:
            score = score + (-1)
        if candidate.text == word_data["poss_adjs"]["female"] and subject.text not in word_data["pronouns"]["subject"]["female"] and subject.text not in word_data["biased_words"]["other_gendered"]["female"]:
            score = score + 1
    if object is not None:
        if object.text in word_data["pronouns"]["object"]["male"] and subject.text not in word_data["pronouns"]["subject"]["male"] and subject.text not in word_data["biased_words"]["other_gendered"]["male"]:
            score = score + (-1)
        if object.text in word_data["pronouns"]["object"]["female"] and subject.text not in word_data["pronouns"]["subject"]["female"] and subject.text not in word_data["biased_words"]["other_gendered"]["female"]:
            score = score + 1
    for candidate in objpron:
        if candidate.text in word_data["pronouns"]["object"]["male"] and subject.text not in word_data["pronouns"]["subject"]["male"] and subject.text not in word_data["biased_words"]["other_gendered"]["male"]:
            score = score + (-1)
        if candidate.text in word_data["pronouns"]["object"]["female"] and subject.text not in word_data["pronouns"]["subject"]["female"] and subject.text not in word_data["biased_words"]["other_gendered"]["female"]:
            score = score + 1
    for tok in sentence:
        if tok.text in word_data["biased_words"]["absolute_bias"]["male"] or tok.text in word_data["biased_words"]["absolute_bias"]["female"]:
            idx = sentence.index(tok)
            quotes_before = 0
            quotes_after = 0
            sent_before = sentence[:idx]
            sent_after = sentence[idx+1:]
            for token in sent_before:
                if token.text in ["\"", "«", "»", "‘", "’"]:
                    quotes_before = quotes_before + 1
            for token in sent_after:
                if token.text in ["\"", "«", "»", "‘", "’"]:
                    quotes_after = quotes_after + 1
            if quotes_before % 2 == 0 and quotes_after % 2 == 0:
                if tok.text in word_data["biased_words"]["absolute_bias"]["male"]:
                    score = score + (-0.5)
                elif tok.text in word_data["biased_words"]["absolute_bias"]["female"]:
                    score = score + 0.5
            else:
                if tok.text in word_data["biased_words"]["absolute_bias"]["male"]:
                    score = score + (-1)
                elif tok.text in word_data["biased_words"]["absolute_bias"]["female"]:
                    score = score + 1
        if tok.pos_ == "VERB":
            if tok.lemma_ in ["mother", "father"]:
                score = score + (-1)
    return score

def identify_gender(entity_text, gencheck):
    entity_text = entity_text.lower()
    gender = "Unspecified"
    itemized = entity_text.split()
    for item in itemized:
        found = np.where(gencheck[:,0] == item)[0].tolist()
        if found != []:
            idx = found[0]
            gender = gencheck[idx,1]
            break
    return gender

def resolve_name(entity_text, pergen, gencheck):
    entity_text_ = entity_text.lower()
    resolved_ = (entity_text_, -1, -1)
    for e in list(pergen['person'].values):
        if e in entity_text_:
            if len(e.split()) == 1 and len(entity_text_.split()) == 2:
                for element in entity_text_.split()[:-1]:
                    if element in gencheck[:,0]:
                        return resolved_
            idx = list(pergen['person'].values).index(e)
            name = list(pergen['resolved_name'].values)[idx]
            isfem = list(pergen['is_female'].values)[idx]
            if isinstance(name, str) and isinstance(isfem, int):
                resolved_ = (name, idx, isfem)
                break
    return resolved_

def pronoun_linkage(doc, prev_prons, gencheck, pergen):
    male_subj, male_obj, female_subj, female_obj = prev_prons[0], prev_prons[1], prev_prons[2], prev_prons[3]
    docnames = list(np.array([e.text.split() for e in doc.ents if e.label_ == "PERSON"]).flatten())
    persons = [item.text for item in doc.ents if item.label_ == "PERSON"]
    persons = [resolve_name(item, pergen, gencheck)[0] for item in persons]
    buffer = ""
    for token in doc:
        #if token.dep_ == "compound" and token.text in docnames:
        if token.dep_ == "compound":
                tokenx = token
                while tokenx.dep_ == "compound":
                    tokenx = tokenx.head
                if tokenx.text in docnames:
                    buffer = buffer + token.text + " "
        else:
                if buffer != "" or token.text in docnames:
                        buffer = buffer + token.text
                        print(buffer)
                        resolved = resolve_name(buffer, pergen, gencheck)[0]
                        if resolved in persons:
                                print(resolved)
                                entgender = identify_gender(resolved, gencheck)
                                if entgender == "Male":
                                        if "subj" in token.dep_:
                                                male_subj = resolved
                                        elif "obj" in token.dep_:
                                                male_obj = resolved
                                elif entgender == "Female":
                                        if "subj" in token.dep_:
                                                female_subj = resolved
                                        elif "obj" in token.dep_:
                                                female_obj = resolved
                buffer = ""
    if male_subj is not None and male_obj is None:
        male_obj = male_subj
    elif male_obj is not None and male_subj is None:
        male_subj = male_obj
    if female_subj is not None and female_obj is None:
        female_obj = female_subj
    elif female_obj is not None and female_subj is None:
        female_subj = female_obj
    print(docnames, persons)
    return male_subj, male_obj, female_subj, female_obj

def pronoun_linkage_v2(doc, prev_prons, gencheck, pergen):
    male_pron, female_pron = prev_prons[0], prev_prons[1]
    docnames = list(np.array([e.text.split() for e in doc.ents if e.label_ == "PERSON"]).flatten())
    persons = [item.text for item in doc.ents if item.label_ == "PERSON"]
    persons = [resolve_name(item, pergen)[0] for item in persons]
    buffer = ""
    for token in doc:
        if token.dep_ == "compound" and token.text in docnames:
                buffer = buffer + token.text + " "
        else:
                if buffer != "":
                        buffer = buffer + token.text
                        if resolve_name(buffer, pergen)[0] in persons:
                                entgender = identify_gender(buffer, gencheck)
                                if entgender == "Male":
                                        male_pron = buffer
                                elif entgender == "Female":
                                        female_pron = buffer
                buffer = ""
    return male_pron, female_pron

def flag_traits(sentence, traitdata):
    score = 0
    fem_traits = traitdata['feminine']
    msc_traits = traitdata['masculine']
    for trait in fem_traits:
        if trait in sentence:
            quotes_before = 0
            quotes_after = 0
            sent_before = sentence[:idx]
            sent_after = sentence[idx+1:]
            for token in sent_before:
                if token.text in ["\"", "«", "»", "‘", "’"]:
                    quotes_before = quotes_before + 1
            for token in sent_after:
                if token.text in ["\"", "«", "»", "‘", "’"]:
                    quotes_after = quotes_after + 1
            if quotes_before % 2 == 0 and quotes_after % 2 == 0:
                score = score + 1
    for trait in msc_traits:
        if trait in sentence:
            quotes_before = 0
            quotes_after = 0
            sent_before = sentence[:idx]
            sent_after = sentence[idx+1:]
            for token in sent_before:
                if token.text in ["\"", "«", "»", "‘", "’"]:
                    quotes_before = quotes_before + 1
            for token in sent_after:
                if token.text in ["\"", "«", "»", "‘", "’"]:
                    quotes_after = quotes_after + 1
            if quotes_before % 2 == 0 and quotes_after % 2 == 0:
                score = score - 1
    return score

def identify_all(doc, pergen, gencheck):
    gendered_names = []
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            tmp = resolve_name(entity.text, pergen, gencheck)
            if tmp != -1:
                gender_cand = identify_gender(entity.text, gencheck)
                if gender_cand == "Male":
                    tmp = 0
                elif gender_cand == "Female":
                    tmp = 1
        tuple = (entity.text, tmp)
        gendered_names.append(tuple)
    return gendered_names
