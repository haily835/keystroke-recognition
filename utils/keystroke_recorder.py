import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import time
import csv
import os
import tkinter.font as tkFont
from tkinter import ttk

typing_data = [
    {
        "title": "The Home Row [j k]",
        "content": "jkj jkj jkj jkj kjk kjk kjk kjk jjj jjj jjj kkk kkk kkk jk jk jk kj kj kj jj kk jk kj kj jk jj jk kk kj j j j j k k k k j k k j j k k j jkj jjk kjj kkj jkk kkk jjj kjk"
    },
    {
        "title": "The Home Row [l ;]",
        "content": "l;l l;l l;l l;l ;l; ;l; ;l; ;l; lll lll lll ;;; ;;; ;;; l; l; l; ;l ;l ;l ll ;; l; ;l ;l l; ll l; ;; ;l l l l l ; ; ; ; l ; ; l l ; ; l l;l ll; ;ll ;;l l;; ;;; lll ;l;"
    },

    {
        "title": "The Home Row [f d]",
        "content": "fff fff fff fff ddd ddd ddd ddd fff ddd fff ddd fff ddd fdf fdf fdf fdf dfd dfd dfd dfd fff fff fff ddd ddd ddd fd fd fd df df df ff dd fd df df fd ff fd dd df"
    },

    {
        "title": "The Home Row [s a]",
        "content": "sss sss sss sss aaa aaa aaa aaa sss aaa sss aaa sss aaa sas sas sas sas asa asa asa asa sss sss sss aaa aaa aaa sa sa sa as as as ss aa sa as as sa ss sa aa as"
    },

    {
        "title": "The Home Row [h g]",
        "content": "hhh hhh hhh hhh ggg ggg ggg ggg hhh ggg hhh ggg hhh ggg hgh hgh hgh hgh ghg ghg ghg ghg hhh hhh hhh ggg ggg ggg hg hg hg gh gh gh hh gg hg gh gh hg hh hg gg gh"
    },


    {
        "title": "The Home Row all together",
        "content": "asa ada afa aga aha aja aka ala a;a ;l; ;k; ;j; ;h; ;g; ;f; ;d; ;s; ;s; ;a; sas sds sfs sgs shs sjs sks sls s;s l;l lkl ljl lhl lgl lfl ldl lsl lal dad dsd dfd dgd dhd dkd dld d;d k;k klk kjk khk kgk kfk kdk ksk kak faf fsf fdf fgf fhf fjf fkf flf f;f f;f jlj jkj jhj jgj jfj jdj jsj jaj gag gsg gdg gfg ghg gkg glg g;g h;h hlh hkh hjh hgh hfh hdh hsh hah a l s j d h g ; f k l f s ; j g k h a d"
    },

    {
        "title": "The top row [u r] with home row",
        "content": "uuu uuu rrr rrr uuu rrr jjj uuu jjj uuu fff rrr fff rrr juj juj frf frf uju uju rfr rfr ara ara srs srs frf frf drd drd grg grg hrh hrh jrj jrj lrl lrl ;r; ;r; aua aua sus sus dud dud fuf fuf gug gug huh huh juj juj kuk kuk lul lul ;u; ;u;"
    },


    {
        "title": "The top row [i e] with home row",
        "content": "iii iii eee eee iii eee kkk iii kkk iii ddd eee ddd eee kik kik ded ded iki iki ede ede aea aea ses ses ded ded fef fef geg geg heh heh jej jej kek kek lel lel ;e; aia aia sis sis did did fif fif gig gig hih hih jij jij kik kik lil lil ;i; ;i;"
    },

    {
        "title": "The top row [o w] with home row",
        "content": "ooo ooo www www ooo www lll ooo lll ooo sss www sss www lol lol sws sws olo olo wsw wsw awa awa sws sws dwd dwd fwf fwf gwg gwg hwh hwh jwj jwj kwk kwk lwl lwl ;w; ;w; aoa aoa sos sos dod dod fof fof gog gog hoh hoh joj joj kok kok lol lol ;o; ;o;"
    },

    {
        "title": "The top row [p g] with home row",
        "content": "ppp ppp qqq qqq ppp qqq ;;; ppp ;;; ppp aaa qqq aaa qqq ;p; ;p; aqa aqa p;p p;p qaq qaq aqa aqa sqs sqs dqd dqd fqf fqf gqg gqg hqh hqh jqj jqj kqk kqk lql lql ;q; ;q; apa apa sps sps dpd dpd fpf fpf gpg gpg hph hph jpj jpj kpk kpk lpl lpl ;p; ;p;"
    },


    {
        "title": "The top row [y t] with home row",
        "content": "yyy yyy ttt ttt yyy ttt ;;; yyy ;;; yyy aaa ttt aaa ttt ;y; ;y; ata ata y;y y;y tat tat ata ata sts sts dtd dtd ftf ftf gtg gtg hth hth jtj jtj ktk ktk ltl ltl ;t; ;t; aya aya sys sys dyd dyd fyf fyf gyg gyg hyh hyh jyj jyj kyk kyk lyl lyl ;y; ;y;"
    },


    {
        "title": "The bottom row [m v] with home row",
        "content": "mmm mmm vvv vvv mmm vvv jjj mmm jjj mmm fff vvv fff vvv jmj jmj fvf fvf mjm mjm vfv vfv ava ava svs svs dvd dvd fvf fvf gvg gvg hvh hvh jvj jvj lvl lvl ;v; ;v; ama ama sms sms dmd dmd fmf fmf gmg gmg hmh hmh jmj jmj kmk kmk lml lml ;m; ;m;"
    },


    {
        "title": "The bottom row [, c] with home row",
        "content": ",,, ,,, ccc ccc ,,, ccc kkk ,,, kkk ,,, ddd ccc ddd ccc k,k k,k dcd dcd ,k, ,k, cdc cdc aca aca scs scs dcd dcd fcf fcf gcg gcg hch hch jcj jcj kck kck lcl lcl ;c; ;c; a,a a,a s,s s,s d,d d,d f,f f,f g,g g,g h,h h,h j,j j,j k,k k,k l,l l,l ;,; ;,;"
    },

    {
        "title": "The bottom row [. x] with home row",
        "content": "... ... xxx xxx ... xxx lll ... lll ... sss xxx sss xxx l.l l.l sxs sxs .l. .l. xsx xsx axa axa sxs sxs dxd dxd fxf fxf gxg gxg hxh hxh jxj jxj kxk kxk lxl lxl ;x; ;x; a.a a.a s.s s.s d.d d.d f.f f.f g.g g.g h.h h.h j.j j.j k.k k.k l.l l.l ;.; ;.;"
    },

    {
        "title": "The bottom row [/ z] with home row",
        "content": "/// /// zzz zzz /// zzz ;;; /// ;;; /// aaa zzz aaa zzz ;/; ;/; aza aza /;/ /;/ zaz zaz aza aza szs szs dzd dzd fzf fzf gzg gzg hzh hzh jzj jzj kzk kzk lzl lzl ;z; ;z; a/a a/a s/s s/s d/d d/d f/f f/f g/g g/g h/h h/h j/j j/j k/k k/k l/l l/l ;/; ;/;"
    },

    {
        "title": "The bottom row [n b] with home row",
        "content": "nnn nnn bbb bbb nnn bbb ;;; nnn ;;; nnn aaa bbb aaa bbb ;n; ;n; aba aba n;n n;n bab bab aba aba sbs sbs dbd dbd fbf fbf gbg gbg hbh hbh jbj jbj kbk kbk lbl lbl ;b; ;b; ana ana sns sns dnd dnd fnf fnf gng gng hnh hnh jnj jnj knk knk lnl lnl ;n; ;n;"
    },

    
    {
        "title": "Right hand capital letters",
        "content": "JJJ KKK LLL JjJ KkK LlL Jjj Kkk Lll HHH HhH Hhh UUU III OOO PPP UuU IiI OoO PpP Uuu Iii Ooo Ppp YYY YyY Yyy MMM  MmM Mmm NNN NnN Nnn"
    },

    {
        "title": "Left hand capital letters",
        "content": "FFF DDD SSS AAA FfF DdD SsS AaA Fff Ddd Sss Aaa GGG GgG Ggg RRR EEE WWW QQQ RrR EeE WwW QqQ Rrr Eee Www Qqq TTT  TtT Ttt VVV CCC XXX ZZZ VvV CcC XxX ZzZ Vvv Ccc Xxx Zzz BBB BbB Bbb"
    },


    {
        "title": "Validate The top row",
        "content": "the four lads stood quietly atop the tower; pear salad is a great quirky dish; plaid dads play golf; slide the glass to your good pal; we gladly yapped for two hours; wade through the water to us; either of us will go; we used our gold goose eggs well; let us do tea for two; go forward to the other side; other ghosts will spook us; the riders had quite a lot of leg power; you see how easy it is to type the top row; a little further to go yet; i wish i had read the flyer fully; read it for us please; joe sipped jade tea jealously; kate flew her fast kite sky high;"
    },

    {
        "title": "Validate 1 The bottom row with words",
        "content": "zebras are not exactly known for being quiet animals. she/he would very likely just play along for a while. you can make quite a lot of lemon zest with even larger sized lemons; please cover each plate. six foxes quickly woke axel; just in time. i caught five jelly fish, six octopus and two sea urchins for my buddy ben. eight / six is exactly equal to four / three."
    },

    {
        "title": "Validate 2 Capital letters",
        "content": "WOW. Four Jacks And One Queen, I Win. Please Do Not Go Up There Now. Zack Needs MORE Caramel Kettle Corn ASAP. Nobody Even HAS Any Purple Vests. Xavier Let Seventy Yellow BEES Into Our Cabin. YIKES. Good Lemons Are HARD To Pick. Do Not Watch Karen; She Is VERY Nervous."
    },

   
    {
        "title": "Test 1 pangrams",
        "content": "The quick brown fox jumps over the lazy dog; Jack packs five dozen liquor jugs. Pack my box with five dozen liquor jugs, and quickly / adjust the bright quartz fox. Jumpy fox skips over black quartz / bright dwarves; lazy dogs bark. Bright vixens jump; dozy fowl quack, lazy dogs bark / quick quartz fixes. Jack big quartz fox skips over lazy dogs; adjust my vow / fix bright liquor jugs."
    },

    {
        "title": "Test 2 pangrams",
        "content": "Quickly adjust the quartz fox, pack my box; bright jugs overflow / lazy dwarves run. Jumping quickly over lazy dogs, the bright fox vexes dwarves; black quartz fills my box / liquor flows. Pack my box with five dozen liquor jugs; quick fox jumps / bright quartz glows. Jacks bright vixen jumps over lazy dogs; dwarves quack, quartz shimmers / bold fox skips. Black quartz, shining brightly, fills my vow; quick fox packs jugs / lazy dogs bark."
    },

    {
        "title": "Test 3 pangrams",
        "content": "The lazy dog jumps; Jacks bright quartz fox skips / over dwarves, packing liquor jugs. Quickly fix the quartz jug; Jacks bright fox jumps / over lazy dogs, vexing dwarves. Jack jumps over the bright quartz; pack my box / lazy dogs bark, quick vixen skips. Dwarves vex quick bright fox; pack my box with liquor jugs, quartz shines / lazy dog jumps. Fix my vow; Jacks quartz fox jumps / quickly over bright lazy dogs, packing liquor jugs. Jumpy dwarves quack; bright fox skips over lazy dogs / quartz fills the liquor jugs, Jack packs."
    },

    {
        "title": "Test 4 email",
        "content": "Good afternoon, The Board of Directors meetings have been scheduled as follows, January, previously confirmed, May, July, and October. Each meeting will take place from 9.00 AM to noon. Further logistical details will be announced closer to each meeting date. If you have any questions, please feel free to contact either Keith Larney or me. Thank you."
    }
]
class TypingTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Test")
        self.root.geometry("800x600")

        # Set a default font for the entire application
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=14)  # Increase the font size

        self.texts_to_type = [lesson["content"] for lesson in typing_data]  # Use content from typing_data
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

        self.video_label = tk.Label(self.frame, font=("Arial", 14, "bold"), bg="#e0e0e0", justify=tk.LEFT)
        self.video_label.pack(anchor=tk.W, pady=5)


        self.label = tk.Label(self.frame, font=("Arial", 14), bg="#e0e0e0", justify=tk.LEFT)
        self.label.pack(anchor=tk.W, pady=5)

        self.display_text = tk.Text(self.frame, height=5, wrap=tk.WORD, state=tk.DISABLED, bg="#f0f0f0", font=("Courier", 15, "bold"))
        self.display_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.typing_area = tk.Text(self.frame, height=5, wrap=tk.WORD, bg="#ffffff", font=("Courier", 15, "bold"))
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

        self.display_text.tag_config("correct", foreground="blue")  # Set color for correct letters
        self.display_text.tag_config("error", foreground="red")  # Set color for error letters

    def update_text_display(self):
        video_file = f'videos/video_{self.current_text_index}.mp4'  # Save video in 'videos' folder
        csv_file = f'labels/video_{self.current_text_index}.csv'      # Save CSV in 'labels' folder
        text_file = f'labels/video_{self.current_text_index}.txt'     # Save TXT in 'labels' folder
        files_exist = os.path.exists(video_file) and os.path.exists(csv_file) and os.path.exists(text_file)

        # Update label with tick mark if files exist
        tick_mark = "✔" if files_exist else ""
        tick_color = "green" if files_exist else "black"
        self.video_label.config(text=f"Text {self.current_text_index + 1}: {tick_mark}", fg=tick_color)
        
        self.display_text.config(state=tk.NORMAL)
        self.display_text.delete(1.0, tk.END)
        self.display_text.insert(tk.END, self.texts_to_type[self.current_text_index])
        self.display_text.config(state=tk.DISABLED)
        self.typing_area.delete(1.0, tk.END)
        self.indicator_label.config(text=f"Text {self.current_text_index + 1} of {len(self.texts_to_type)}")

        # Update to show the current lesson title and intro
        current_lesson = typing_data[self.current_text_index]
        self.label.config(text=current_lesson['title'])  # Update label with title and intro

    def log_key(self, event):
        if not self.recording:
            return
        
        key = event.keysym.lower()

        if key == 'shift_l' or key == 'shift_r' or key == 'caps_lock':
            key = 'shift'
        
        self.all_timings[self.current_text_index].append((key, self.frame_count))
        print(f"Key: {key}, Frame Count: {self.frame_count}")

        # Highlight the typed text
        typed_text = self.typing_area.get(1.0, tk.END).strip()  # Get the current typed text
        original_text = self.texts_to_type[self.current_text_index]  # Get the original text
        self.display_text.config(state=tk.NORMAL)  # Enable editing to apply tags

        # Clear previous highlights
        self.display_text.delete(1.0, tk.END)  # Clear the display text
        self.display_text.insert(tk.END, original_text)  # Insert the original text

        # Highlight the typed text
        # Text index in tkinter https://tkdocs.com/tutorial/text.html#modifying
        for i, char in enumerate(typed_text):
            if i < len(original_text):
                if char == original_text[i]:
                    self.display_text.tag_add("correct", f"1.0+{i}c", f"1.0+{i+1}c")  # Highlight correct letters in blue
                else:
                    self.display_text.tag_add("error", f"1.0+{i}c", f"1.0+{i+1}c")  # Highlight incorrect letters in red

        self.display_text.config(state=tk.DISABLED)  # Disable editing again

    def clear_text(self):
        # Clear the typing area and timings
        self.all_timings[self.current_text_index] = []
        self.typing_area.delete(1.0, tk.END)
        
        print(f"Cleared text {self.current_text_index + 1}")

        # Delete video and CSV files if they exist
        video_file = f'videos/video_{self.current_text_index}.mp4'
        csv_file = f'labels/video_{self.current_text_index}.csv'
        text_file = f'labels/video_{self.current_text_index}.txt'  # Define the text file path
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
            self.display_text.config(state=tk.NORMAL)
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
            csv_file = f'labels/video_{self.current_text_index}.csv'      # Save CSV in 'labels' folder
            text_file = f'labels/video_{self.current_text_index}.txt'     # Save TXT in 'labels' folder

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
