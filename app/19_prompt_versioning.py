from langsmith import traceable
from config import llm_open_router
from langchain import hub

# Sample: https://smith.langchain.com/hub/glisteningocean/gais-email-outreach

# Vary the version number to see different results as we continuously improve on our outreach prompt
version = 0

prompt_versions = [
    "glisteningocean/gais-email-outreach:87377fd7",
    "glisteningocean/gais-email-outreach:e5508f2c",
    "glisteningocean/gais-email-outreach:bf1bae4a",
    "glisteningocean/gais-email-outreach:8e1b682e",
]

recipient = """
Name: John Smith
Email: john@smith.com

Personal blog: https://smith.com/blog
Latest blog posts:
- https://smith.com/blog/2023/10/10/extracting-exif-data-from-images
- https://smith.com/blog/2023/10/05/using-ai-to-generate-image-captions
- https://smith.com/blog/2023/09/15/innovative-photography-techniques-using-the-latest-tech-gadgets
- https://smith.com/blog/2023/09/10/starting-a-photography-business-the-tech-tools-you-need
Latest github commits:
- chore: added new post
- feat: installed session analytics
- fix: stripe payment bug
- feat: added AI tool for photo captioning
"""

@traceable
def main():
    prompt = hub.pull(prompt_versions[version])
    
    runnable = prompt | llm_open_router

    response = runnable.invoke({
        "recipient": recipient
    })
    print(response.content)

if __name__ == "__main__":
    main()
