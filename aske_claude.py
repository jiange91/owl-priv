import anthropic
import dotenv

dotenv.load_dotenv()

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
)

message = client.messages.create(
    temperature=0.0,
    max_tokens=1024,
    model="claude-sonnet-4-20250514",
    messages=[
        {"role": "user", "content": "A standard Rubik\u2019s cube has been broken into cubes making up its sides. The cubes are jumbled, and one is removed. There are 6 cubes with one colored face, 12 edge cubes with two colored faces, and 8 corner cubes with three colored faces. All blue cubes have been found. All cubes directly left, right, above, and below the orange center cube have been found, along with the center cube. The green corners have all been found, along with all green that borders yellow. For all orange cubes found, the opposite face\u2019s cubes have been found. The removed cube has two colors on its faces. What are they? Answer using a comma separated list, with the colors ordered alphabetically."}
    ]
)
print(message.content)