
def get_inspired_item_text(item, db_dict):
    if item.item_id in db_dict:
        return db_dict[item.item_id]
    return item.item_id


def get_lavic_item_text(item):
    parts = []
    if item.name:
        parts.append(item.name)
    if 'category' in item.attributes:
        parts.append(item.attributes['category'])
    if 'description' in item.attributes:
        desc = item.attributes['description']
        if isinstance(desc, list):
            parts.extend(desc)
        else:
            parts.append(desc)
    if 'features' in item.attributes:
        features = item.attributes['features']
        if isinstance(features, list):
            parts.extend(features[:5])
    if 'details' in item.attributes and 'Brand' in item.attributes['details']:
        parts.append(item.attributes['details']['Brand'])
    if 'price' in item.attributes and item.attributes['price']:
        parts.append(item.attributes['price'])
    if 'average_rating' in item.attributes and item.attributes['average_rating']:
        parts.append(item.attributes['average_rating'])
    text = ' '.join(str(p) for p in parts if p)
    return text if text else item.item_id


def get_lavic_conversation_text(conversation):
    if 'context' in conversation:
        return conversation['context']
    return ""


def batch_get_item_texts(items, dataset, db_dict=None):
    item_texts = {}
    for item_id, item in items.items():
        if dataset == "inspired":
            text = get_inspired_item_text(item, db_dict)
        elif dataset == "lavic":
            text = get_lavic_item_text(item)
        else:
            text = item_id
        item_texts[item_id] = text
    return item_texts
