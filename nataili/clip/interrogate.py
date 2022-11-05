import clip
import torch
from nataili.util import logger
from nataili.util.voodoo import performance


class Interrogator:
    def __init__(self, model, preprocess, data_lists, device, batch_size=100):
        self.model = model
        self.preprocess = preprocess
        self.data_lists = data_lists
        self.device = device
        self.batch_size = batch_size

    def rank(self, model, image_features, text_array, device, top_count=2):
        top_count = min(top_count, len(text_array))
        text_tokens = clip.tokenize([text for text in text_array]).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = torch.zeros((1, len(text_array))).to(device)
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]

        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
        return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy() * 100)) for i in range(top_count)]

    def batch_rank(self, model, image_features, text_array, device):
        batch_size = min(self.batch_size, len(text_array))
        batch_count = int(len(text_array) / batch_size)
        batches = []
        for i in range(batch_count + 1):
            batches.append(text_array[i * batch_size : (i + 1) * batch_size])
        if len(text_array) % batch_size != 0:
            batches.append(text_array[batch_count * batch_size :])
        ranks = []
        for batch in batches:
            ranks += self.rank(model, image_features, batch, device)
        return ranks

    @performance
    def __call__(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        ranks = []
        bests = [[("", 0)]] * 7
        logger.info("Ranking text")
        ranks.append(self.batch_rank(self.model, image_features, self.data_lists["mediums"], self.device))
        ranks.append(self.batch_rank(self.model, image_features, self.data_lists["flavors"], self.device))
        ranks.append(self.batch_rank(self.model, image_features, self.data_lists["artists"], self.device))
        ranks.append(self.batch_rank(self.model, image_features, self.data_lists["movements"], self.device))
        ranks.append(self.batch_rank(self.model, image_features, self.data_lists["sites"], self.device))
        ranks.append(self.batch_rank(self.model, image_features, self.data_lists["techniques"], self.device))
        ranks.append(self.batch_rank(self.model, image_features, self.data_lists["tags"], self.device))
        logger.info("Sorting text")
        for i in range(len(ranks)):
            confidence_sum = 0
            for ci in range(len(ranks[i])):
                confidence_sum += ranks[i][ci][1]
            if confidence_sum > sum(bests[i][t][1] for t in range(len(bests[i]))):
                bests[i] = ranks[i]

        for best in bests:
            best.sort(key=lambda x: x[1], reverse=True)

        medium = []
        for m in bests[0][:1]:
            medium.append({"text": m[0], "confidence": m[1]})
        artist = []
        for a in bests[1][:2]:
            artist.append({"text": a[0], "confidence": a[1]})
        trending = []
        for t in bests[2][:2]:
            trending.append({"text": t[0], "confidence": t[1]})
        movement = []
        for m in bests[3][:2]:
            movement.append({"text": m[0], "confidence": m[1]})
        flavors = []
        for f in bests[4][:2]:
            flavors.append({"text": f[0], "confidence": f[1]})
        techniques = []
        for t in bests[5][:2]:
            techniques.append({"text": t[0], "confidence": t[1]})
        tags = []
        for t in bests[6][:2]:
            tags.append({"text": t[0], "confidence": t[1]})
        return {
            "medium": medium,
            "artist": artist,
            "trending": trending,
            "movement": movement,
            "flavors": flavors,
            "techniques": techniques,
            "tags": tags,
        }
