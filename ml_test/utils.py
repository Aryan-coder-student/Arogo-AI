def clean_gender(gender):
    gender = str(gender).strip().lower()
    male_variants = ["m", "male", "man", "cis male", "cis man", "maile", "mal", "malr", "male (cis)", "guy (-ish) ^_^", "male-ish", "maile" ,"msle", "mail", "malr"]
    female_variants = ["f", "female", "woman", "cis female", "cis-female/femme", "female (cis)", "femake", "trans woman", "female (trans)"]
    non_binary_variants = [
        "non-binary", "genderqueer", "fluid", "androgyne", "agender", "enby", "queer/she/they", "something kinda male?",
        "ostensibly male, unsure what that really means", "male leaning androgynous"
    ]
    
    if gender in male_variants:
        return "Male"
    elif gender in female_variants:
        return "Female"
    elif gender in non_binary_variants:
        return "Non-binary"
    elif gender in {"nah", "all", "a little about you", "p"}:
        return "Unknown"
    else:
        return "Other"
