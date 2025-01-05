import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import time
import csv
import os
import tkinter.font as tkFont
from tkinter import ttk

class TypingTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Test")
        self.root.geometry("800x600")

        # Set a default font for the entire application
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=14)  # Increase the font size

        self.texts_to_type = [
        "The quick brown fox jumps over 123 lazy dogs five times. On January 1st, 2025, 7 zebras danced gracefully, eating 89 pizzas under 6 bright lamps. Curious owls, 4 total, observed as 3 frogs croaked near the 0 degree pond. Bees buzzed 12 times before spotting 23 flowers growing in a vibrant A to Z rainbow. Xylophones, said the quirky parrots, make a joyful sound every 5 to 9 seconds.Meanwhile, 345 sailors sang sea shanties beneath 678 twinkling stars, their voices echoing 890 chords into the vast abyss.",

        "Amid the vibrant festival on June 5th, 2024, 7 lively musicians played tunes from A to Z. Crowds cheered as 8 zebras dashed past 9 glowing lanterns, while 345 balloons floated above the scene. The quick fox jumped over 12 lazy dogs 6 times, barking loudly each time. An owl hooted 3 times as 0 frogs hopped along the muddy path, observing the scattered petals of a rainbow colored bouquet. By midnight, 89 twinkling stars illuminated the stage, and 567 partygoers sang in harmony, clapping and shouting numbers like 1, 2, 3, and 4 with unbridled joy, celebrating life to the fullest.",

        "On a crisp October evening, 5 daring adventurers set out to explore the forest, carrying maps marked from A to Z. They discovered 12 hidden caves, 8 sparkling streams, and 9 glowing fireflies that lit up the night. The quick brown fox dashed past 7 lazy squirrels, while 3 owls hooted loudly from 5 towering trees. Nearby, a group of 6 frogs leaped over 0 rocks into a pond that shimmered like silver under the moonlight. As they camped, 345 stars twinkled above, and they counted 678 fireflies circling around, marveling at the magical beauty of numbers and nature entwined.",

        "Beneath the bright evening sky, 5 young explorers gathered to count the stars, each taking turns naming constellations from A to Z. The quick brown fox leapt gracefully over 7 sleepy dogs, while 3 chirping crickets serenaded the night. In the distance, 12 owls hooted, and 6 frogs splashed in a pond reflecting 89 twinkling stars. Around the campfire, they shared stories of the 345 adventures they would dreamed of, punctuated by laughter and the sound of 0 worries in the world. By midnight, they had counted 678 glowing fireflies and marveled at how nature, like numbers, always found a way to bring magic to life.",

        "In the bustling town square, 5 jugglers performed tricks with glowing balls marked A to Z, dazzling a crowd of 345 cheering onlookers. A quick brown fox darted between 7 colorful stalls, while 6 frogs croaked melodiously by a 0 degree fountain. Nearby, 12 musicians played a lively tune, their instruments creating a symphony of sound. As the clock struck 8, 9 firecrackers lit up the sky, and 89 children clapped in delight. The energy of the night was electric, with 678 stars twinkling above and laughter echoing from all corners, celebrating the unbreakable rhythm of life, letters, and numbers.",

        "On a breezy summer afternoon, 5 adventurers trekked through a vibrant meadow, their backpacks labeled from A to Z. A quick brown fox dashed past 7 grazing deer, while 6 frogs hopped near a glimmering stream. Overhead, 9 birds circled the sky, and 12 dragonflies danced above 0 ripples in the water. Nearby, 345 flowers bloomed in radiant colors, attracting 89 buzzing bees. As dusk fell, the group gazed at 678 stars lighting up the heavens, marveling at the beauty of nature's endless patterns and the harmony between the alphabet and numbers in their journey.",

        "At dawn, 5 curious children wandered into the enchanted forest, guided by a trail marked with letters A through Z. The quick brown fox playfully darted between 7 towering trees, while 6 frogs leapt into a 0 degree crystal clear pond. Overhead, 9 sparrows chirped as they flew past 12 shimmering cobwebs glistening in the light. The adventurers counted 89 mushrooms growing in a spiral and spotted 345 fireflies blinking in rhythmic harmony. By nightfall, under a sky boasting 678 glittering stars, they shared stories, blending the magic of letters and numbers into an unforgettable tale.",

        "In 2023, 758925 scientists worked on 512355 experiments for 6515135 months, running 151808 tests daily across 166437 labs with 4900 computers and 300 machines each processing 769310 datasets. Over 563260 prototypes were developed, 421230 errors were fixed, and 100 equations were solved. By day 35680, they reduced 9 major problems to 0, scaling the project to impact 75770 cities using 537 formulas and 9689 reactors. Their 9632449 budget supported 10 teams across 4670 locations, benefiting 116895 homes with renewable energy in under 26356 months, while collecting over 5263632 data points",

        "Under the warm glow of 678 stars, 5 travelers ventured through a lively marketplace filled with signs from A to Z. A quick brown fox darted past 7 bustling stalls selling everything from 12 jars of honey to 89 vibrant scarves. Nearby, 6 frogs croaked in harmony beside a fountain at 0 degrees, its waters sparkling in the moonlight. Children laughed as they counted 345 lanterns lighting up the square, while 9 musicians played a cheerful tune. The night buzzed with energy, a perfect blend of numbers, letters, and the boundless joy of life.",

        "As the sun set over the hills, 5 hikers gathered around a campfire, their backpacks embroidered with letters from A to Z. A quick brown fox darted past 7 chirping crickets and 6 frogs leaping near the 0 degree lake. Overhead, 9 bats flitted through the twilight, while 12 fireflies blinked in rhythm among the trees. The hikers shared stories of their 89 mile journey, marveling at the beauty of the 345 wildflowers blooming in the valley. Under the glow of 678 stars, they laughed and embraced the harmony of letters, numbers, and natures wonders.",

        "On a cool spring evening, 5 friends explored a magical garden where flowers bloomed in patterns from A to Z. A quick brown fox dashed between 7 rows of roses, while 6 frogs sang near a pond chilled to 0 degrees. Overhead, 9 swallows darted gracefully, and 12 glowing butterflies danced in the air. The group counted 89 shining crystals embedded in a nearby fountain and marveled at 345 twinkling lights strung through the trees. As the clock struck midnight, 678 stars sparkled above, casting a celestial glow over the enchanted scene",

        "In the heart of a mystical valley, 5 explorers uncovered a hidden realm where the alphabet from A to Z adorned ancient stone carvings. A quick brown fox sprinted past 7 glittering waterfalls, while 6 frogs perched on lily pads in a pond shimmering at 0 degrees. Overhead, 9 luminous birds soared, their wings reflecting the light of 12 golden orbs. Among the lush greenery, 89 blooming orchids stretched skyward, and 345 fireflies danced in perfect unison. Beneath the canopy of 678 twinkling stars, the adventurers marveled at the harmony of letters, numbers, and the magic of the universe.",

        "On a starlit night, 5 stargazers gathered on a hill, their notebooks filled with sketches from A to Z. A quick brown fox darted across the meadow, weaving between 7 curious rabbits and 6 frogs croaking beside a 0 degree brook. Overhead, 9 shooting stars streaked through the sky as 12 owls hooted softly from nearby trees. The group marveled at 89 constellations they had mapped and the 345 glimmering lights of a distant village. Surrounded by the brilliance of 678 celestial bodies, they felt the seamless connection between the infinite letters, numbers, and the wonders of the cosmos.",

        "At the break of dawn, 5 adventurers set off on a trail marked by signposts labeled A to Z. A quick brown fox scurried past 7 colorful butterflies fluttering near 6 frogs basking on the edge of a 0 degree stream. In the distance, 9 deer grazed peacefully under the shade of 12 ancient oaks. The travelers paused to admire 89 wildflowers blooming in vibrant hues and counted 345 pebbles forming a natural mosaic along the path. By nightfall, under a sky glittering with 678 stars, they shared stories of discovery, marveling at the unity of natures letters and numbers.",


        "Good afternoon, The Board of Directors meetings have been scheduled as follows, January, previously confirmed, May, July, and October. Each meeting will take place from 9.00 AM to noon. Further logistical details will be announced closer to each meeting date. If you have any questions, please feel free to contact either Keith Larney or me. Thank you.",

        "Dear Dr. Lay, I am pleased to invite you to join my CERA colleagues and me at our annual executive conference, Shooting the Rapids, Strategies and Risks for the Energy Future, to be held this February in Houston, Texas. This premier international gathering offers senior executives new ideas, insights, and strategic thinking on the challenges facing the global energy industry. The conference will focus on the implications of current market turmoil for energy strategies, investment, regulatory challenges, competitive dynamics, and industry structure, with presentations covering oil, natural gas, power, and their interconnections across key regions. As the centerpiece of CERAWeek, it also provides opportunities for informal interaction and networking, with last year event drawing executives from over 50 countries. For more details, please visit CERA.com or register at register.cera.com. I hope you will join us. Sincerely, Daniel Yergin Chairman, Cambridge Energy Research Associates",
        ]
        self.current_text_index = 0
        self.all_timings = [[] for _ in self.texts_to_type]
        self.media_recorder = None
        self.recording = False

        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def show_about(self):
        messagebox.showinfo("About", "Typing Test Application\nVersion 1.0")

    def create_widgets(self):
        self.frame = tk.Frame(self.root, padx=10, pady=10, relief=tk.RAISED, borderwidth=2, bg="#e0e0e0")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.label = tk.Label(self.frame, font=("Arial", 14, "bold"), bg="#e0e0e0")
        self.label.pack(anchor=tk.W, pady=5)

        self.display_text = tk.Text(self.frame, height=5, wrap=tk.WORD, state=tk.DISABLED, bg="#f0f0f0", font=("Arial", 14))
        self.display_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.typing_area = tk.Text(self.frame, height=5, wrap=tk.WORD, bg="#ffffff", font=("Arial", 14))
        self.typing_area.pack(fill=tk.BOTH, expand=True, pady=5)
        self.typing_area.bind("<KeyRelease>", self.log_key)

        self.button_frame = tk.Frame(self.frame, bg="#e0e0e0")
        self.button_frame.pack(fill=tk.X, pady=5)

        # Use default tkinter buttons without custom styles
        self.start_btn = tk.Button(self.button_frame, text="Step 1: Start Recording", command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(self.button_frame, text="Step 3: Stop Recording", command=self.stop_recording)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_text)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.back_btn = tk.Button(self.button_frame, text="Back", command=self.previous_text)
        self.back_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(self.button_frame, text="Next", command=self.next_text)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        self.indicator_label = tk.Label(self.frame, text="", font=("Arial", 10), bg="#e0e0e0")
        self.indicator_label.pack(anchor=tk.E, pady=5)

        self.update_text_display()

    def update_text_display(self):
        video_file = f'videos/video_{self.current_text_index}.mp4'  # Save video in 'videos' folder
        csv_file = f'labels/video-{self.current_text_index}.csv'      # Save CSV in 'labels' folder
        text_file = f'labels/video-{self.current_text_index}.txt'     # Save TXT in 'labels' folder
        files_exist = os.path.exists(video_file) and os.path.exists(csv_file) and os.path.exists(text_file)

        # Update label with tick mark if files exist
        tick_mark = "✔" if files_exist else ""
        tick_color = "green" if files_exist else "black"
        self.label.config(text=f"Text {self.current_text_index + 1}: {tick_mark}", fg=tick_color)
        
        self.display_text.config(state=tk.NORMAL)
        self.display_text.delete(1.0, tk.END)
        self.display_text.insert(tk.END, self.texts_to_type[self.current_text_index])
        self.display_text.config(state=tk.DISABLED)
        self.typing_area.delete(1.0, tk.END)
        self.indicator_label.config(text=f"Text {self.current_text_index + 1} of {len(self.texts_to_type)}")

    def log_key(self, event):
        key = event.keysym.lower()
        self.all_timings[self.current_text_index].append((key, self.frame_count))
        print(f"Key: {key}, Frame Count: {self.frame_count}")

    def clear_text(self):
        # Clear the typing area and timings
        self.all_timings[self.current_text_index] = []
        self.typing_area.delete(1.0, tk.END)
        print(f"Cleared text {self.current_text_index + 1}")

        # Delete video and CSV files if they exist
        video_file = f'videos/video_{self.current_text_index}.mp4'
        csv_file = f'labels/video-{self.current_text_index}.csv'
        text_file = f'labels/video-{self.current_text_index}.txt'  # Define the text file path
        if os.path.exists(video_file):
            os.remove(video_file)
            print(f"Deleted {video_file}")
        if os.path.exists(csv_file):
            os.remove(csv_file)
            print(f"Deleted {csv_file}")
        if os.path.exists(text_file):  # Check if the text file exists
            os.remove(text_file)  # Delete the text file
            print(f"Deleted {text_file}")  # Print confirmation

        # Update display to remove tick mark
        self.update_text_display()

    def next_text(self):
        if self.current_text_index < len(self.texts_to_type) - 1:
            self.current_text_index += 1
            self.update_text_display()

    def previous_text(self):
        if self.current_text_index > 0:
            self.current_text_index -= 1
            self.update_text_display()

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.start_time = int(time.time() * 1000)  # Record the start time in milliseconds
            self.frame_count = 0  # Initialize frame count

            # Define folder paths
            videos_folder = 'videos'
            labels_folder = 'labels'
            ground_truth_folder = 'ground_truth'  # New folder for ground truth

            # Create folders if they do not exist
            os.makedirs(videos_folder, exist_ok=True)
            os.makedirs(labels_folder, exist_ok=True)
            os.makedirs(ground_truth_folder, exist_ok=True)  # Create ground truth folder

            # Update video filename to include the folder path
            video_filename = f'{videos_folder}/video_{self.current_text_index}.mp4'
            self.media_recorder = cv2.VideoWriter(
                video_filename,
                cv2.VideoWriter_fourcc(*'mp4v'),
                30.0,  # Assuming 30 FPS
                (640, 480)  # Assuming a resolution of 640x480
            )
            print(f"Recording started: {video_filename}")

            # Start a new thread to capture and write frames
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.start()

    def capture_frames(self):
        cap = cv2.VideoCapture(0)  # Open the default camera
        if not cap.isOpened():  # Check if the camera is connected
            print("Error: Camera not connected.")
            return  # Exit the function if the camera is not connected
        while self.recording:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                self.media_recorder.write(frame)  # Write the frame to the video file
                self.frame_count += 1  # Increment frame count
            else:
                break
        cap.release()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.capture_thread.join()  # Wait for the capture thread to finish
            self.media_recorder.release()
            print("Recording stopped")

            # Calculate and print FPS
            end_time = int(time.time() * 1000)  # Get end time in milliseconds
            duration = (end_time - self.start_time) / 1000.0  # Convert duration to seconds
            fps = self.frame_count / duration if duration > 0 else 0
            print(f"Recorded FPS: {fps:.2f}")

            # Define file paths
            video_file = f'videos/video_{self.current_text_index}.mp4'  # Save video in 'videos' folder
            csv_file = f'labels/video-{self.current_text_index}.csv'      # Save CSV in 'labels' folder
            text_file = f'labels/video-{self.current_text_index}.txt'     # Save TXT in 'labels' folder

            # Save key timings to CSV
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Key', 'Frame'])  # Write header
                for key, milliseconds in self.all_timings[self.current_text_index]:
                    writer.writerow([key, milliseconds])
            print(f"Key timings saved to {csv_file}")

            # Save typed text to a text file
            with open(text_file, mode='w') as text_file:
                typed_text = self.typing_area.get(1.0, tk.END).strip()  # Get the typed text
                text_file.write(typed_text)  # Write the typed text to the file
            print(f"Typed text saved to {text_file}")

            self.update_text_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = TypingTestApp(root)
    root.mainloop()
