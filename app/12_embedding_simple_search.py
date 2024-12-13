from config import hf_embeddings
from scipy.spatial.distance import cosine
import numpy as np

document = [
    "I learned being a solo SaaS founder isnt for me. Backstory- Im 24 and have ran probably 5+ businesses so far. I cant code and have always done service/product businesses. Never SaaS Recently i wanted to do a SaaS as I thought i had seen a big market opportunity. I posted in a discord that i was looking to get the MVP made, and after some interviews i found my person. So far it’s been great and hes done a phenomenal job. Today i asked the dev how much in salary would he need to do this full time. He said he cant’t because he had his own startup. This truthfully fucked me up because i am so used to being useful. I had to accept that being a solo SaaS founder would likely be the WORST thing i could be. Imagine me hiring a software engineer and then me, the ONLY founder, sitting in my office not writing a single piece of code. I’d lose my entire engineering team. Should I keep trying to do service/product businesses or bust my ass to find a technical co founder who wants to do this with me?",
    "How much are you spending on marketing? which channels? I am curious to learn how much you are spending on marketing as a SaaS startup? Which channels are you utilizing and how is the response from each one? At our startup, we are doing blog posts, guest posts, and partly Google Ads. All in all about $500 - $1k spend per month for generating new leads. We get a lot of clicks from Google (5% CTR) but actual conversion is quite slow and in many cases non-existent.",
    "At what point a gpt wrapper is no longer a gpt wrapper?, I am currently working on a SaaS, and the product is based on communicating back and forth with about 10 fine-tuned gpt models. I'm looking forward to adding some other models such as claude or gemini. It doesn't really matter what my startup does (I don't know either yet 100% lol), but I was just thinking, at what point something isn't a gpt wrapper anymore?",
    "Community Heroes: How I bring environmental issues to life in classrooms and help students cultivate a love for nature. Singapore has about 450 active ground-up initiatives, made up of groups of individuals who come together in self-organised projects to help the community. TODAY's Voices section is publishing first-hand accounts of young changemakers and the stories behind their initiatives.  Here, Ms Cassandra Yip-Lee, 24, recounts the inspiration behind her initiative, Earth School, the challenges faced in championing environmental education in Singapore and her plans for the future.",
    "Gen Zen: Meditation apps are a dime a dozen. I tried out 5 of them for a week and here's what I found. Increasingly, people are becoming aware of the importance of mental health and well-being in our lives. In our weekly Gen Zen series, TODAY looks at ways that we can feel better while coping with the mental stresses of modern life.",
    "As temperatures rise, low-income families in rental units struggle to beat the heat. While all Singaporeans are coping with hotter weather, low-income communities are especially vulnerable, with cooling solutions like air-conditioning often out of reach. As temperatures rise, some rental residents said they turn to makeshift, creative solutions like spraying ice water on the walls or showering up to four times a day to stay cool. The heat has taken a physical, mental and financial toll on many, as some said their electricity and water bills have doubled or the heat has directly affected their income",
    "The Big Read: S'poreans lack hunger, can't compete because they want more work-life balance? Not true and here's why. A yearning for better work-life balance and flexible work arrangements has increasingly become a norm in Singapore. However, this has also reignited debate over whether these are too much to ask for, with some employers saying that they may even rethink hiring local staff. Workers and experts interviewed by TODAY laid out some factors behind the growing demand for work-life harmony and debunked the notion that this speaks of a workforce which lacks hunger or is uncompetitive. Employers meanwhile spoke about the challenges they face in navigating this new landscape",
    "'No choice but to allow it': Parents cite challenges in keeping devices away from kids amid concerns over excessive screen time. Deputy Prime Minister Lawrence Wong recently announced plans for having additional safeguards around excessive screen time for children. This came amid a rise in mental health concerns worldwide among the younger generation. Experts say that media use for children under two years old has been linked to developmental issues. TODAY speaks to eight parents on their personal experiences and challenges in limiting screen time for their children",
]


def main():
    my_query = "Climate change"

    embedding = hf_embeddings.embed_query(my_query)

    # Calculate embeddings for all documents
    document_embeddings = [hf_embeddings.embed_query(doc) for doc in document]

    # Calculate cosine similarity between query embedding and each document embedding
    similarities = [1 - cosine(embedding, doc_emb) for doc_emb in document_embeddings]

    # Get the indices of the top 2 most similar documents
    top_2_indices = np.argsort(similarities)[
        -2:
    ]  # This gets the last two indices, which have the highest similarity

    # Print the top 2 most similar documents
    for index in reversed(top_2_indices):  # reversed to print the most similar first
        print(f"Document {index + 1}: {document[index]}")
        print(f"Similarity: {similarities[index]}\n")


if __name__ == "__main__":
    main()
