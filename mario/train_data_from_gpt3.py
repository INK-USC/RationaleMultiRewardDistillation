# to sample from gpt-3
import os
import openai
import argparse
import pandas as pd
import json
from collections import defaultdict

print("Here1")

def get_args():
    print("in args")
    parser = argparse.ArgumentParser(description='calling GPT-3')

    # gpt-3 stuff
    parser.add_argument(
        '--temperature', type=float, default=0.1, help='temperature for sampling.')
    parser.add_argument(
        '--arch', type=str, default='davinci-instruct-beta', help='gpt-3 arch to use')

    # datasets stuff
    parser.add_argument(
        '--dataset-name', type=str, default='strategyqa', help='name of dataset')
    parser.add_argument(
        '--dataset-train', type=str, default='',
        help='JSONL file containing train data. Each row must contain a prompt at `row["question"]`.')
    parser.add_argument(
        '--dataset-val', type=str, default='',
        help='JSONL file containing val data. Each row must contain a prompt at `row["question"]`.')
    parser.add_argument(
        '--dataset-test', type=str, default='',
        help='JSONL file containing test data. Each row must contain a prompt at `row["question"]`.')
    parser.add_argument(
        '--which_split', type=str, default='test', help='which split to run')
    parser.add_argument(
        '--gen-mode', type=str, default='i2ro', help='name of gen_mode out of i2o, i2ro')
    parser.add_argument(
        '--use_demonstrations', type=int, default=1, help='0/1 whether to use demonstrations')

    # saving stuff
    parser.add_argument(
        '--save_dir', type=str, default="", help="where to save the outputs")

    args = parser.parse_args()
    return args

def get_demonstrations(dataset_name = 'obqa', gen_mode = 'i2ro'):
    # defines the demonstrations for the datasets in different gen-mode settings
    demonstrations = {}

    # penguin
    demonstrations['penguins_in_a_table'] = {}
    demonstrations['penguins_in_a_table']['i2ro'] = ["Q: Here is a table where the first line is a header and each subsequent line is a penguin: name, age, height (cm), weight (kg)\nLouis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15 For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm. We now add a penguin to the table:\nJames, 12, 90, 12\nHow many penguins are less than 8 years old?\nOptions:\n(A) 1\n(B) 2\n(C) 3\n(D) 4\n(E) 5\nA: Let's think step by step.\nThis question focuses on age. We know the following: Louis is 7 years old, Bernard is 5 years old, Vincent is 9 years old, and Gwen is 8 years old. Now, we add James to this table: James is 12 years old. The penguins that are less than 8 years old are Louis and Bernard. There are 2 penguins less than 8 years old. So the answer is (B).\n\nQ: Here is a table where the first line is a header and each subsequent line is a penguin: name, age, height (cm), weight (kg)\nLouis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15 For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm. Which is the youngest penguin?\nOptions:\n(A) Louis\n(B) Bernard\n(C) Vincent\n(D) Gwen\n(E) James\nA: Let's think step by step.\nThis question focuses on age. We know the following: Louis is 7 years old, Bernard is 5 years old, Vincent is 9 years old, and Gwen is 8 years old. According to the table, Bernard (5) is the youngest amongst them. The youngest penguin is Bernard. So the answer is (B).\n\nQ: Here is a table where the first line is a header and each subsequent line is a penguin: name, age, height (cm), weight (kg)\nLouis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15 For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm. What is the name of the second penguin sorted by alphabetic order?\nOptions:\n(A) Louis\n(B) Bernard\n(C) Vincent\n(D) Gwen\n(E) James\nA: Let's think step by step.\nThis question focuses on the name. We know the following: The names of the penguin in the table are Louis, Bernard, Vincent, and Gwen. When we sort their names alphabetically, we get Bernard, Gwen, Louis, Vincent. The name of the second penguin sorted by alphabetical order is Gwen. The name of the second penguin sorted by alphabetic order is Gwen. So the answer is (D).\n\nQ:"]
    demonstrations['penguins_in_a_table']['i2ro'].append("\nA: Let's think step by step.\n")

    # strategyqa
    demonstrations['strategyqa'] = {}
    demonstrations['strategyqa']['i2o'] = ["Q: Do hamsters provide food for any animals?\nA: So the answer is yes.\n\nQ: Could Brooke Shields succeed at University of Pennsylvania?\nA: So the answer is yes.\n\nQ: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?\nA: So the answer is no.\n\nQ: Yes or no: Is it common to see frost during some college commencements?\nA: So the answer is yes.\n\nQ: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?\nA: So the answer is no.\n\nQ: Yes or no: Would a pear sink in water?\nA: So the answer is no.\n\nQ: Yes or no: "]
    demonstrations['strategyqa']['i2ro'] = ["Q: Do hamsters provide food for any animals? Hamsters are prey animals. Prey animals provide food for predators.\nA: So the answer is yes.\n\n \
                                    Q: Could Brooke Shields succeed at University of Pennsylvania? Brooke Shields graduated from Princeton University. Princeton is ranked as the number 1 national college by US news. University of Pennsylvania is ranked as number 6 national college by US news. Princeton only admits around 6 percent of applicants as of 2018. University of Pennsylvania accepts around 9% of applicants as of 2018.\nA: So the answer is yes.\n\n\
                                    Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls? Hydrogen is the first element and has an atomic number of one. To square a number, you multiply it by itself. The Spice Girls has five members.\nA: So the answer is no.\n\n\
                                    Q: Yes or no: Is it common to see frost during some college commencements? College commencement ceremonies often happen during the months of December, May, and sometimes June.  Frost isn't uncommon to see during the month of December, as it is the winter.\nA: So the answer is yes.\n\n\
                                    Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)? The War in Vietnam (1945-46) lasted around 6 months. The gestation period for a llama is 11 months.\nA: So the answer is no.\n\n\
                                    Q: Yes or no: Would a pear sink in water? The density of a raw pear is about 0.59 g\/cm^3. The density of water is about 1 g\/cm^3. Objects only sink if they are denser than the surrounding fluid.\nA: So the answer is no.\n\n\
                                    Q: Yes or no: "]

    # obqa
    demonstrations['obqa'] = {}
    demonstrations['obqa']['i2ro'] = ["Q: Poison causes harm to which of the following? (a) a Tree (b) a robot (c) a house (d) a car\nA: Poison will harm living things, only a tree is a living thing. So the answer is: (a)\n\nQ: As you look deeper into a Marbel you can see (a) the future (b) minut defects (c) colors (d) the other side\nA: Marbel is not transparent, so you can not see the other side. Marbel does not necessarily have multiple colors. You will see minut defects. So the answer is: (b)\n\nQ: When food is reduced in the stomach (a) the mind needs time to digest (b) take a second to digest what I said (c) nutrients are being deconstructed (d) reader's digest is a body of works\nA: The food is being deconstructed in the stomach during digestion. So the answer is: (c)\n\nQ: The sun is responsible for (a) puppies learning new tricks (b) children growing up and getting old (c) flowers wilting in a vase (d) plants sprouting, blooming and wilting\nA: The sun can affect the growing of living things, like plants. So the answer is: (d)\n\nQ: "]
    demonstrations['obqa']['i2ro'].append("\nA: ")
    demonstrations['obqa']['i2ro_new'] = ["Q: The sun is responsible for (a) puppies learning new tricks (b) children growing up and getting old (c) flowers wilting in a vase (d) plants sprouting, blooming and wilting\nA: A plant requires sunlight for photosynthesis, which accumulates resources required for sprouting, blooming and wilting. So the answer is: (d)\n\nQ: When standing miles away from Mount Rushmore  (a) the mountains seem very close (b) the mountains are boring (c) the mountains look the same as from up close (d) the mountains seem smaller than in photographs\nA: When an object is far away, it takes up less of your field of view, and so seems smaller than in the photographs. So the answer is: (d)\n\nQ: When food is reduced in the stomach (a) the mind needs time to digest (b) take a second to digest what I said (c) nutrients are being deconstructed (d) reader's digest is a body of works\nA: The stomach is part of the digestive system. The breaking down of food into nutrients occurs in the digestive system. So the answer is: (c)\n\nQ: Poison causes harm to which of the following? (a) a Tree (b) a robot (c) a house (d) a car\nA: A tree is a living thing. Poison causes harm to living things. So the answer is: (a)\n\nQ: A magnet will stick to (a) a belt buckle (b) a wooden table (c) a plastic cup (d) a paper plate\nA: A belt buckle is made of metal. If a magnet is attracted to a metal then that magnet will stick to that metal. So the answer is: (a)\n\nQ: Deer are less safe in the woods because wolves (a) have fur (b) howl (c) have claws (d) have tails\nA: Claws are used by wolves to catch prey like deer. So the answer is: (c)\n\nQ: An electric car causes (a) more CO2 emissions (b) equal CO2 emissions (c) electric emissions (d) less CO2 emissions\nA: An electric car uses less gasoline than a regular car and thus causes less CO2 emissions. So the answer is: (d)\n\nQ:"]
    demonstrations['obqa']['i2ro_new'].append("\nA: ")

    # quarel
    demonstrations['quarel'] = {}
    demonstrations['quarel']['i2ro'] = ["Q: Mike was snowboarding on the snow and hit a piece of ice. He went much faster on the ice because _____ is smoother. (A) snow (B) ice\nA: When something is smoother, it is easier to slide on. Thus, he could go faster on the ice because ice is smoother. So the answer is: (B)\n\nQ: I could hear then boy that say close to me clear as day, however I could not hear the young lady sitting in the back of the room.  WHo am I able to hear louder (A) Boy (B) Lady\nA: When someone is close, it is easier to hear them. I also could not hear the young lady well. Thus, I am able to hear the boy louder. So the answer is: (A)\n\nQ: I watched the snowflakes go from tiny specks in the sky to a nice size once they fell on my face.  When did the snowflakes seem bigger (A) in the sky (B) on my face\nA: When something is closer, it seems bigger. The snowflakes is closer when they are on my face. Thus, they seem bigger when they are on my face. So the answer is: (B)\n\nQ: When Tammy tried to slide the glass mixing bowl down the marble counter top to her mom, it came to a dead stop when it reached the wooden cutting board. The bowl came to a stop because the wooden cutting board has (A) more resistance or (B) less resistance\nA: When something has more resistance, it is harder to slide. Thus, the bowl came to a stop because the wooden cutting board has more resistance. So the answer is: (A)\n\nQ: Sarah walked through the city and saw a tourist attraction she wanted to visit. She had several blocks to go to get to it, and the attraction looked very small. As she got close it though, it towered over her. This is because when she was close to it the attraction looked (A) much bigger (B) much smaller.\nA: When something is closer, it looks bigger. Thus, the attraction looked much bigger when she was close to it. So the answer is: (A)\n\nQ: "]
    demonstrations['quarel']['i2ro'].append("\nA: ")

    # csqa
    demonstrations['csqa'] = {}
    demonstrations['csqa']['i2ro'] = ["Q: What do people use to absorb extra ink from a fountain pen?\nAnswer Choices:\n(a) shirt pocket\n(b) calligrapher's hand\n(c) inkwell\n(d) desk drawer\n(e) blotter\nA: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is: (e)\n\nQ: What home entertainment equipment requires cable?\nAnswer Choices:\n(a) radio shack\n(b) substation\n(c) cabinet\n(d) television\n(e) desk\nA: The answer must require cable. Of the above choices, only television requires cable. So the answer is: (d)\n\nQ: The fox walked from the city into the forest, what was it looking for?\nAnswer Choices:\n(a) pretty flowers.\n(b) hen house\n(c) natural habitat\n(d) storybook\n(e) dense forest\nA: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is: (c)\n\nQ: Sammy wanted to go to where the people were.  Where might he go?\nAnswer Choices:\n(a) race track\n(b) populated areas\n(c) the desert\n(d) apartment\n(e) roadblock\nA: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is: (b)\n\nQ: Where do you put your grapes just before checking out?\nAnswer Choices:\n(a) mouth\n(b) grocery cart\n(c) super market\n(d) fruit basket\n(e) fruit market\nA: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is: (b)\n\nQ: Google Maps and other highway and street GPS services have replaced what?\nAnswer Choices:\n(a) united states\n(b) mexico\n(c) countryside\n(d) atlas\n(e) oceans\nA: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is: (d)\n\nQ: Before getting a divorce, what did the wife feel who was doing all the work?\nAnswer Choices:\n(a) harder\n(b) anguish\n(c) bitterness\n(d) tears\n(e) sadness\nA: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is: (c)\n\nQ: "]
    demonstrations['csqa']['i2ro'].append("\nA: ")
    demonstrations['csqa']['i2ro_new'] = ["Q: What do people use to absorb extra ink from a fountain pen?\nAnswer Choices:\n(a) shirt pocket\n(b) calligrapher's hand\n(c) inkwell\n(d) desk drawer\n(e) blotter\nA: Blotting paper absorbs liquids like ink well. So the answer is: (e)\n\nQ: What home entertainment equipment requires cable?\nAnswer Choices:\n(a) radio shack\n(b) substation\n(c) cabinet\n(d) television\n(e) desk\nA: Cable can be fixed to a television. Television is a home entertainment equipment. So the answer is: (d)\n\nQ: The fox walked from the city into the forest, what was it looking for?\nAnswer Choices:\n(a) pretty flowers.\n(b) hen house\n(c) natural habitat\n(d) storybook\n(e) dense forest\nA: Forests are one of the main natural habitats of foxes. So the answer is: (c)\n\nQ: Sammy wanted to go to where the people were.  Where might he go?\nAnswer Choices:\n(a) race track\n(b) populated areas\n(c) the desert\n(d) apartment\n(e) roadblock\nA: Populated areas are where there are a lot of people. So the answer is: (b)\n\nQ: Where do you put your grapes just before checking out?\nAnswer Choices:\n(a) mouth\n(b) grocery cart\n(c) super market\n(d) fruit basket\n(e) fruit market\nA: Grocery cart is used in stores by customers to collect purchases. Checking out of purchases is done in checkout area of stores. So the answer is: (b)\n\nQ: Google Maps and other highway and street GPS services have replaced what?\nAnswer Choices:\n(a) united states\n(b) mexico\n(c) countryside\n(d) atlas\n(e) oceans\nA: Atlas are replaced by more precise Google maps, other highway and street GPS services. One can get much more precise data with the help of Google maps and Street GPS services. So the answer is: (d)\n\nQ: Before getting a divorce, what did the wife feel who was doing all the work?\nAnswer Choices:\n(a) harder\n(b) anguish\n(c) bitterness\n(d) tears\n(e) sadness\nA: Bitterness is the resentful feeling of anger at being treated unfairly. Doing all the work means being treated unfairly. So the answer is: (c)\n\nQ: "]
    demonstrations['csqa']['i2ro_new'].append("\nA: ")
    demonstrations['csqa']['i2ro_again'] = ['Q: Where can you find a dogs house? Answer Choices: (a) found outside (b) faithful (c) frightening (d) cold (e) four legs\nA: A dog needs a house to protect it from the weather and predators. The house should be big enough for the dog to move around in, but not too big or the dog will be cold. The house should have a door to keep out the cold and a roof to keep out the rain. The house should be made of sturdy materials that will not collapse in the wind. The best place to find a dog house is at a pet store or a home improvement store. You can also find dog houses online. So the answer is: (a).\n\nQ: Where would you find a ticket booth and see a concert? Answer Choices: (a) clerk (b) indoors (c) movie theater (d) venue (e) auditorium\nA: You would go to a ticket booth to purchase a ticket for a concert. The concert would take place at a venue, which could be indoors or outdoors. It is most likely that the concert would take place indoors at a large auditorium or a movie theater. So the answer is: (d).\n\nQ: What do people aim to do at work? Answer Choices: (a) complete job (b) learn from each other (c) kill animals (d) wear hats (e) talk to each other\nA: People usually go to work in order to earn money, which they can use to buy things they want or need. In order to do this, they must complete their job. This usually means working with others in order to get the job done efficiently. Sometimes, people learn new things from their coworkers. For example, they might learn a new skill or how to do their job better. Finally, people usually talk to their coworkers during their break or after work. So the answer is: (a).\n\nQ: Where do you store a large container? Answer Choices: (a) supermarket (b) factory (c) juice (d) hostel (e) cabinet\nA: We need to store a large container. It won’t fit in a cabinet, so that’s not the answer. It also won’t fit in a juice container, so that’s not the answer. It could fit in a factory, but that’s not necessarily where you would store it. It could also fit in a supermarket, but that’s not necessarily where you would store it. The most likely answer is a hostel. So the answer is: (d).\n\nQ: If there is a pond with trees around it, where it it likely located? Answer Choices: (a) ground (b) bathroom (c) forest (d) countryside (e) rural area\nA: A pond is likely to be located near trees because trees need water to survive. Therefore, the most likely location for a pond with trees around it is in a forest. So the answer is: (c).\n\nQ: James was meeting a friend. They had planed a slow day. They didn’t want to do much. They just wanted what? Answer Choices: (a) relaxation (b) panic (c) alarm (d) joy (e) cheer\nA: James was meeting a friend. They had planned a slow day. They didn’t want to do much. They just wanted _____. The only logical answer choice is relaxation. So the answer is: (a).\n\nQ: what does a person do to stay healthy? Answer Choices: (a) fever (b) eat every day (c) excited (d) headache (e) expressive\nA: To stay healthy, a person should probably do things like eating healthy foods, exercising, and getting enough rest. So the answer choices that best fit are (b) eat every day, (c) excited, and (d) headache. So the answer is: (b).\n\nQ:']
    demonstrations['csqa']['i2ro_again'].append("\nA: ")

    # wg
    demonstrations['wg'] = {}
    demonstrations['wg']['i2ro'] = ["Q: The GPS and map helped me navigate home. I got lost when the _ got turned off. \n (A) GPS (B) map\nA: A GPS device is electronic, while a map is paper-based. So the answer is: (A)\n\nQ: I picked up a bag of peanuts and raisins for a snack. I wanted a sweeter snack out so I ate the _ for now. \n (A) raisins (B) peanuts\nA: Peanuts contain a lot of fat. Raisins contain a lot of sugar. So the answer is: (A)\n\nQ: The geese prefer to nest in the fields rather than the forests because in the _ predators are more hidden. \n (A) fields (B) forests\nA: There are more trees in the forests than in the fields. So the answer is: (B)\n\nQ: Once in Poland, Dennis enjoyed the trip more than Jason because _ had a shallow understanding of the Polish language. \n (A) Dennis (B) Jason\nA: Those who know the native language would enjoy the trip better. So the answer is: (B)\n\nQ: Adam put handwash only clothes in the washer but Aaron washed them by hand as _ was lazy. \n (A) Adam (B) Aaron\nA: Washing clothes with washer takes less effort than by hand. So the answer is: (A)\n\nQ:"]
    demonstrations['wg']['i2ro'].append("\nA: ")

    # qasc
    demonstrations['qasc'] = {}
    demonstrations['qasc']['i2ro'] = ["Q: How do you reduce pollution? \nAnswer choices: (A) igniting fuel and oxidiser (B) transportation technology (C) wasting (D) not recycling (E) burning fossil fuels (F) converting electricity to heat (G) water conservation (H) using less resources \nA: Conserving resources has a positive impact on the environment. Use of resources affects the environment such as pollution. So the answer is: (H)\n\nQ: what will move to another area if their habitat will no longer support them?\nAnswer choices: (A) density (B) Birds (C) squids (D) humans (E) clouds (F) gravity (G) cows (H) Whales\nA: If a habitat can no longer support animals then those animals will move to another area. Cows are social animals. So the answer is: (G)\n\nQ: With the exception of allergies, what may cause a person to seek medical attention?\nAnswer choices: (A) Contact with latex (B) a tree falling (C) Organs within the body. (D) Contact with baby chicks (E) prolactin release (F) Contact with peanut butter (G) hypothyroidism (H) Contact with microorganisms\nA: Microorganisms can cause infections. Infections usually require medical treatment. So the answer is: (H)\n\nQ: Lavender can induce\nAnswer choices: (A) healing (B) energy (C) hormones (D) mutations (E) Heart rate (F) growth (G) symptoms (H) warmth\nA: Healing requires rest. Lavender induces restful sleep. So the answer is: (A)\n\nQ: what state is a liquid in when frozen?\nAnswer choices: (A) vapor (B) dense (C) gas (D) cooled (E) steam (F) solid (G) boiling (H) cold\nA: Freezing means changing from a liquid into a solid by reducing heat energy. Liquids freeze when they change to the solid state. So the answer is: (F)\n\nQ: what unites to form a diploid zygote?\nAnswer choices: (A) plant reproduction (B) Most plants (C) orchids (D) sperm and ova (E) salt and pepper (F) predator and prey (G) honeybees (H) diploids and zygotes\nA: Gametes then unite in fertilization and form a diploid zygote. Collectively, the sperm and the ova are also referred to as gametes. So the answer is: (D)\n\nQ: What absorbs all visible light?\nAnswer choices: (A) apples (B) coal (C) Green (D) coral (E) skin (F) bamboo (G) glass (H) eyes\nA: If an object is black then that object absorbs all visible light. Light grains are quartz, Black grains are coal. So the answer is: (B)"]
    demonstrations['qasc']['i2ro'].append("\nA: ")

    # numersense
    demonstrations['numersense'] = {}
    demonstrations['numersense']['i2ro'] = ["Q: penguins have <mask> wings. \n (A) no (B) zero (C) one (D) two (E) three (F) four (G) five (H) six (I) seven (J) eight (K) nine (L) ten\nA: Birds have two wings. Penguin is a kind of bird. So the answer is (D).\n\nQ: a parallelogram has <mask> sides. \n (A) no (B) zero (C) one (D) two (E) three (F) four (G) five (H) six (I) seven (J) eight (K) nine (L) ten\nA: A rectangular is a parallelogram. A square is a parallelogram. So the answer is (F).\n\nQ: there are <mask> feet in a yard. \n (A) no (B) zero (C) one (D) two (E) three (F) four (G) five (H) six (I) seven (J) eight (K) nine (L) ten\nA: A yard is three feet. So the answer is (E).\n\nQ: water can exist in <mask> states. \n (A) no (B) zero (C) one (D) two (E) three (F) four (G) five (H) six (I) seven (J) eight (K) nine (L) ten\nA: There states for matter are solid, liquid, and gas. So the answer is (E).\n\nQ: a typical human being has <mask> limbs. \n (A) no (B) zero (C) one (D) two (E) three (F) four (G) five (H) six (I) seven (J) eight (K) nine (L) ten\nA: Human has two arms and two legs. So the answer is (F)\n\nQ: "]
    demonstrations['numersense']['i2ro'].append("\nA: ")

    # coin flip
    demonstrations['coinflip'] = {}
    demonstrations['coinflip']['i2ro'] = ["Q: A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up?\nA: The coin was flipped by Ka and Sherrie. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up. So the answer is yes.\n\nQ: A coin is heads up. Jamey flips the coin. Teressa flips the coin. Is the coin still heads up?\nA: The coin was flipped by Jamey and Teressa. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up. So the answer is yes.\n\nQ: A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin. Is the coin still heads up?\nA: The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no.\n\nQ: A coin is heads up. Millicent does not flip the coin. Conception flips the coin. Is the coin still heads up?\nA: The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no.\n\nQ: A coin is heads up. Sal flips the coin. Raymond does not flip the coin. Is the coin still heads up?\nA: The coin was flipped by Sal. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no.\n\nQ: A coin is heads up. Conception flips the coin. Kristian does not flip the coin. Is the coin still heads up?\nA: The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no.\n\nQ: A coin is heads up. Inga does not flip the coin. Elanor does not flip the coin. Is the coin still heads up?\nA: The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, and it was not flipped, so it is still heads up. So the answer is yes.\n\nQ: A coin is heads up. Ryan flips the coin. Shaunda flips the coin. Is the coin still heads up?\nA: The coin was flipped by Ryan and Shaunda. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up. So the answer is yes"]
    demonstrations['coinflip']['i2ro'].append("\nA: ")

    # last letter 
    demonstrations['lastletter'] = {}
    demonstrations['lastletter']['i2ro'] = ['Q: Take the last letters of the words in "Elon Musk" and concatenate them.\nA: The last letter of "Elon" is "n". The last letter of "Musk" is "k". Concatenating them is "nk". So the answer is nk.\n\nQ: Take the last letters of the words in "Larry Page" and concatenate them.\nA: The last letter of "Larry" is "y". The last letter of "Page" is "e". Concatenating them is "ye". So the answer is ye.\n\nQ: Take the last letters of the words in "Sergey Brin" and concatenate them.\nA: The last letter of "Sergey" is "y". The last letter of "Brin" is "n". Concatenating them is "yn". So the answer is yn.\n\nQ: Take the last letters of the words in "Bill Gates" and concatenate them.\nA: The last letter of "Bill" is "l". The last letter of "Gates" is "s". Concatenating them is "ls". So the answer is ls.\n\nQ:']
    demonstrations['lastletter']['i2ro'].append("\nA: ")

    # sports understanding
    demonstrations['sports_understanding'] = {}
    demonstrations['sports_understanding']['i2ro'] = ['Q: Is the following sentence plausible? “Kyle Palmieri was called for slashing.”\nA: Kyle Palmieri is a hockey player. Being called for slashing is part of hockey. So the answer is yes. \n\nQ: Is the following sentence plausible? “Joao Moutinho caught the screen pass in the NFC championship.”\nA: Joao Moutinho is a soccer player. The NFC championship is part of American football, not soccer. So the answer is no. \n\nQ: Is the following sentence plausible? “Carson Wentz set the pick and roll.”\nA: Carson Wentz is an American football player. Pick and roll is part of basketball, not football. So the answer is no. \n\nQ: Is the following sentence plausible? “Jonas Valanciunas beat the buzzer.”\nA: Jonas Valanciunas is a basketball player. Beating the buzzer is part of basketball. So the answer is yes. \n\nQ: Is the following sentence plausible? “Jamel Murray was perfect from the line.”\nA: Jamal Murray is a basketball player. Being perfect from the line is part of basketball. So the answer is yes. \n\nQ: Is the following sentence plausible? “Sam Darnold passed the puck.”\nA: Sam Darnold is a American football player. Passing the puck is part of hockey, not American football. So the answer is no. \n\nQ: Is the following sentence plausible? “Draymond Green threw a touchdown.”\nA: Draymond Green is an basketball player. Throwing a touchdown is part of football, not basketball. So the answer is no. \n\nQ: Is the following sentence plausible? “Malcolm Brogdon banked the shot in.”\nA: Malcolm Brogdon is a basketball player. Banking the shot in is part of basketball. So the answer is yes.\n\nQ:']
    demonstrations['sports_understanding']['i2ro'].append("\nA: ")
    return demonstrations

class PromptDataset:
    def __init__(self, path, dataset_name, gen_mode, use_demonstrations):
        demonstrations = get_demonstrations(dataset_name = dataset_name)

        if dataset_name == 'numersense':
            self.data = pd.read_csv(path, delimiter='\t', header=None)
            self.data.columns = ["question", "answer"]
        else:
            self.data = pd.read_json(path, orient='records')
            if dataset_name in ["lastletter"]:
                pass
            elif dataset_name!= "strategyqa":
                self.data = self.data.T
        
        print(self.data.head)
        self.prompts = []
        self.questions = []
        self.references = []
        self.gold_labels = []

        if dataset_name=="strategyqa":
            classes = {0 : "no", 1 : "yes"}
        elif dataset_name == "numersense":
            # classes = {(A): "no", (B): "zero", (C): "one", (D): "two", (E): "three",\
            #             (F): "four", (G): "five", (H): "six", (I): "seven", (J): "eight",\
            #             (K): "nine", (L): "ten"}
            classes = {"no": "(A)", "zero": "(B)", "one": "(C)", "two": "(D)", "three": "(E)",\
                        "four": "(F)", "five": "(G)", "six": "(H)", "seven": "(I)", "eight": "(J)",\
                        "nine": "(K)", "ten": "(L)"}
        if use_demonstrations:
            ct_demonstrations = demonstrations[dataset_name][gen_mode]
        for i in range(len(self.data)):
            d = self.data.iloc[i]
            ct_question = d['question'] + " \n (A) no (B) zero (C) one (D) two (E) three (F) four (G) five (H) six (I) seven (J) eight (K) nine (L) ten"
            self.questions.append(ct_question)
            # ct_rationale = d['rationale']
            # if ct_rationale[-1]!='.': ct_rationale = ct_rationale + '.'
            
            if dataset_name in ["strategyqa", "numersense"]:
                ct_label = classes[d['answer']]
            else:
                ct_label = d['answer']

            # adding demonstrations
            if use_demonstrations:
                if len(ct_demonstrations) == 1:
                    ct_question = ct_demonstrations[0] + ct_question
                else:
                    ct_question = ct_demonstrations[0] + ct_question + ct_demonstrations[1]

            self.prompts.append(ct_question)
            # self.gold_labels.append('So the answer is ' + ct_label + '.')
            self.gold_labels.append(ct_label.lower())
            # if gen_mode=="i2ro":
            #     self.references.append(ct_rationale + ' The answer is ' + ct_label + '.')
            # elif gen_mode=="i2o":
            #     self.references.append('The answer is ' + ct_label + '.')
            
            if i < 2:
                print("Question:", self.questions[i])
                print("Prompt:", self.prompts[i])
                # print("Reference:", self.references[i])
                print("Gold label sequence:", self.gold_labels[i])

                print("\n\n\n")


openai.api_key=''

def main():
    print("HI HI HI HI")
    args = get_args()

    # save dir stuff
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, args.dataset_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f'Write to output directory: {args.save_dir}')

    if args.which_split == "train":
        dataset = PromptDataset(args.dataset_train, args.dataset_name, args.gen_mode, args.use_demonstrations)
    if args.which_split == "val":
        dataset = PromptDataset(args.dataset_val, args.dataset_name, args.gen_mode, args.use_demonstrations)
    if args.which_split == "test":
        dataset = PromptDataset(args.dataset_test, args.dataset_name, args.gen_mode, args.use_demonstrations)
    print("Loaded dataset " + args.which_split + ":", len(dataset.prompts))

    # answer template stuff
    if args.dataset_name in ['strategyqa', 'coinflip', 'lastletter', 'sports_understanding']:
        lll = ["The answer is", "So the answer is", "So, the answer is", "Thus the answer is", "Thus, the answer is", "Of the choices, the most appropriate answer is"]
    else:
        lll = ["The answer is (", "So the answer is (", "So, the answer is (", "Thus the answer is (", "Thus, the answer is (", "Of the choices, the most appropriate answer is (", "The correct answer is ("]
    lll_lower = [ct_lll.lower() for ct_lll in lll]

    if args.dataset_name in ['strategyqa', 'coinflip', 'lastletter', 'sports_understanding']:
        lll_lower.remove("the answer is") # since this will be there in all of them, it has to be the last thing we resort to
        lll_lower.append("the answer is")
    else:
        lll_lower.remove("the answer is (") # since this will be there in all of them, it has to be the last thing we resort to
        lll_lower.append("the answer is (")

    if args.dataset_name in ['strategyqa', 'coinflip', 'lastletter', 'sports_understanding']:
        answer_template = "the answer is"
    else:
        answer_template = "the answer is ("


    # calling gpt-3
    gpt3_response_jsonl = {}
    running_idx = 0
    acc_count = 0
    good_responses = []
    accuracy_dict1 = defaultdict(list)
    accuracy_dict2 = {}
    unique_questions = []
    for i in range(len(dataset.prompts)):
        print("i:", i)
        #if i==5: break
        ct_prompt = dataset.prompts[i]
        #ct_len = 2048 - len(ct_prompt)
        #print("current length at i=" + str(i) + ":", ct_len)
        ct_gold_label = dataset.gold_labels[i]
        ct_question = dataset.questions[i]

        if args.which_split=="train":
            if ct_question in unique_questions:
                print("question already done:", ct_question)
                continue
            unique_questions.append(ct_question)

        for ii in range(1):
            while True:
                try:
                    ct_response = openai.Completion.create(
                        engine=args.arch,
                        prompt=ct_prompt,
                        max_tokens=100,
                        temperature=args.temperature)
                except:
                    continue
                ct_response_text = ct_response['choices'][0]['text']
                ct_response_text = ct_response_text.replace("\n", " ").lstrip()
                print(ct_response_text)

                if len(ct_response_text) == 0 or ct_response_text.startswith("Q:"):
                    continue
                if answer_template not in ct_response_text.replace(':', '').lower():
                    continue

                tmp = []
                for ct_lll in lll + lll_lower:
                    if ct_lll in ct_response_text.replace(':', ''):
                        tmp = ct_response_text.replace(':', '').split(ct_lll)
                        break

                if len(tmp) == 0:
                    continue

                ct_predicted_rationale = tmp[0].strip().rstrip(" A")
                if args.dataset_name not in ['strategyqa', 'coinflip', 'lastletter', 'sports_understanding']:
                    ct_predicted_label = '(' + tmp[1].strip()[:2].lower()
                elif args.dataset_name in ['strategyqa', 'coinflip', 'sports_understanding']:
                    if tmp[1].strip().lower().startswith('no'):
                        ct_predicted_label = 'no'
                    elif tmp[1].strip().lower().startswith('yes'):
                        ct_predicted_label = 'yes'
                    else:
                        ct_predicted_label = ""
                elif args.dataset_name in ['lastletter']:
                    ct_predicted_label = tmp[1].strip().lower()[:2] # because the training data has only 2 letters
                if ct_predicted_label != "":     
                    good_responses.append(i)

                break

            # saving
            gpt3_response_jsonl[running_idx] = {'question': ct_question,\
            'gold_label': ct_gold_label, 'predicted_label': ct_predicted_label,\
            'predicted_rationale': ct_predicted_rationale}
            accuracy_dict1[ct_question].append(ct_predicted_label)
            accuracy_dict2[ct_question] = ct_gold_label


            if running_idx%1 == 0:
                print('\n---------------------------------------------------------------------------------------------------------------')
                print(i)
                #print('ct_prompt:', ct_prompt)
                print("Question:", ct_question)
                print("GPT-3's response:", ct_response_text.replace("\n", " "))
                print("pred label:", ct_predicted_label)
                print("pred rationale:", ct_predicted_rationale)
            running_idx += 1

            # if ct_predicted_label==ct_gold_label:
            #     acc_count += 1
            # else:
            #     print("wrong answer:", ct_predicted_label, ct_gold_label)

    with open(os.path.join(args.save_dir, args.which_split + "_gpt3_responses.jsonl"), "w") as f:
        json.dump(gpt3_response_jsonl, f)
    #print("% valid responses:", float(running_idx)*100/i)
    print("% valid responses:", len(good_responses), running_idx, len(good_responses)/running_idx*100)

    acc_count = 0
    for q in accuracy_dict1:
        majority_pred_label = max(set(accuracy_dict1[q]), key = accuracy_dict1[q].count)
        gold_label = accuracy_dict2[q]
        if majority_pred_label==gold_label:
            acc_count += 1
    print("accuracy:", acc_count/len(accuracy_dict1)*100)

if __name__=="__main__":
    main()
