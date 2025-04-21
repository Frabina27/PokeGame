import requests
import random
import os
import cv2
import numpy as np
import time
import mediapipe as mp
from PIL import Image, ImageEnhance

# ========== PokeAPI CONFIGURATION ==========
POKEMON_API = "https://pokeapi.co/api/v2/pokemon/"
IMG_DIR = "assets"
CAUGHT_LIST = []


# ========== FETCHING A RANDOM POKEMON ==========

def fetch_random_pokemon():
    
    poke_id = random.randint(1,151) 
    response = requests.get(f"{POKEMON_API}{poke_id}")
    
    data = response.json()
    
    
    name = data["name"]
    img_url = data["sprites"]["other"]["official-artwork"]["front_default"]
    return name, img_url


# ========== DOWNLOAD IMAGE ==========

def download_image(url, name):
    img_path = f"assets/{name}.png"
    # Create folder if it doesn't exist
    os.makedirs(IMG_DIR, exist_ok=True)
    
    # Fetch Image from API
    img_data = requests.get(url).content
    
    # Save image to file
    with open(img_path, 'wb') as f:
        f.write(img_data)
    
    return img_path 


# ========== CREATE SILHOUETTE IMAGE USING PILLOW ==========

def create_silhouette(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGBA")
    
    # Convert the image to grayscale
    grayscale = image.convert("L")
    
    # Merge to keep transparency while making it black
    silhouette = Image.merge("RGBA", (grayscale, grayscale, grayscale, image.split()[-1]))
    
    # Darken the image completely to make it fully black (silhouette effect)
    enhancer = ImageEnhance.Brightness(silhouette)
    silhouette = enhancer.enhance(0)
    
    # Save silhouette image
    sil_path = image_path.replace(".png", "_silhouette.png")
    silhouette.save(sil_path)
    
    return sil_path


# ========== SHOW IMAGE USING OPENCV ==========

def show_image_cv2(path, title="Image"):
    # Read the image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img.shape[2] == 4:
        # Add white background for transparent images
        bg = np.ones(img.shape, dtype=np.uint8) * 255
        alpha = img[:, :, 3] / 255.0
        for c in range(3):
            bg[:, :, c] = (1 - alpha) * bg[:, :, c] + alpha * img[:, :, c]
        img = bg.astype(np.uint8)
    
    # Show the image in a window
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ========== GAME: CATCHING POKEMON WITH HAND MOTION ==========

def catch_pokemon_with_webcam(pokemon_name):
    print("Show your hand and move it UP or DOWN to throw the Pokeball!")

    # Setup MediaPipe Hands for detecting hand landmarks
    mp_hands = mp.solutions.Hands #Hand tracking
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    cap = cv2.videoCapture(0) #Opens webcam

    y_positions = []  # Track Y position of palm
    tracking_started = False  # Whether hand tracking has started
    pokeball_shown = False  # Whether Pokeball is visible
    caught = None  # Result of the catch
    start_time = None  # When tracking started
    result_shown_time = None  # When catch result was shown

    # Load Pokeball image or draw a fallback
    try:
        pokeball_img = cv2.imread("pokeball.png", cv2.IMREAD_UNCHANGED)
        if pokeball_img is None:
            raise FileNotFoundError
    except:
        # Create fallback Pokeball image if file not found
        pokeball_img = np.zeros((100, 100, 4), dtype=np.uint8)
        cv2.circle(pokeball_img, (50, 50), 45, (0, 0, 255, 255), -1)  # Red half
        cv2.circle(pokeball_img, (50, 50), 45, (255, 255, 255, 255), 2)  # Border
        cv2.line(pokeball_img, (10, 50), (90, 50), (255, 255, 255, 255), 2)  # Center line
        cv2.circle(pokeball_img, (50, 50), 10, (255, 255, 255, 255), -1)  # Center button

    while cap.isOpened():
        success, frame = cap.read()  # Read frame from webcam
        if not success:
            continue

        frame = cv2.flip(frame, 1)  # Mirror the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        h, w = frame.shape[:2]

        # Detect hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            if not tracking_started:
                tracking_started = True
                start_time = time.time()  # Start tracking timer

            tracking_duration = time.time() - start_time

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate palm center based on landmarks
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                mid_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                palm_x = int((wrist.x + mid_base.x) * w / 2)
                palm_y = int((wrist.y + mid_base.y) * h / 2)

                # Track hand movement vertically
                y_positions.append(palm_y)
                
                if len(y_positions) > 30:
                    y_positions.pop(0)

                # Wait for 5 seconds of stable tracking before showing Pokeball
                if tracking_duration >= 5 and not pokeball_shown:
                    pokeball_shown = True
                    print("Your pokeball is ready! Move your hand UP/DOWN to throw!")

                if pokeball_shown and caught is None:
                    # Draw Pokeball image on the user's palm
                    pb_size = 100
                    pb = cv2.resize(pokeball_img, (pb_size, pb_size))
                    x_offset = max(0, min(palm_x - pb_size // 2, w - pb_size))
                    y_offset = max(0, min(palm_y - pb_size // 2, h - pb_size))
                    alpha = pb[:, :, 3] / 255.0

                    for c in range(3):
                        frame[y_offset:y_offset+pb_size, x_offset:x_offset+pb_size, c] = (
                            alpha * pb[:, :, c] +
                            (1 - alpha) * frame[y_offset:y_offset+pb_size, x_offset:x_offset+pb_size, c]
                        )

                    # Detect throw gesture by analyzing hand movement
                    if len(y_positions) >= 15:
                        # Measure change in Y
                        delta = y_positions[-1] - y_positions[0]
                        
                        # Big movement(meaning a large change in Y) = throwing action of the ball
                        if abs(delta) > 50:
                            
                            # 75% chance to catch
                            caught = random.random() < 0.75
                            print(f"🎉 Good job! You caught {pokemon_name.title()}!" if caught else "😢 Uh oh.. The Pokemon got away...")
                            result_shown_time = time.time()

        # Show instructions and result on screen, while CV is open
        # putText - (frame, text, position, styling)
        if tracking_started and not pokeball_shown:
            remaining = max(0, 5 - (time.time() - start_time))
            cv2.putText(frame, f"Tracking... {remaining:.1f}s to Pokeball", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif pokeball_shown and caught is None:
            cv2.putText(frame, "Throw the Pokeball!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        elif caught is not None:
            result_msg = "Caught!" if caught else "Got away!"
            color = (0, 255, 0) if caught else (0, 0, 255)
            cv2.putText(frame, result_msg, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.putText(frame, "Press Q to quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Pokemon Catch Game", frame)

        key_pressed = cv2.waitKey(1) & 0xFF
        quit_pressed = key_pressed == ord('q')
        time_after_result = result_shown_time and time.time() - result_shown_time > 5

        if quit_pressed or time_after_result:
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    if caught:
        CAUGHT_LIST.append(pokemon_name.title())


# ========== MAIN GAME LOOP ==========

def main():
    print("Welcome to Who's That Pokemon?\n")

    pokemon_name, img_url = fetch_random_pokemon()
    img_path = download_image(img_url, pokemon_name)
    silhouette_path = create_silhouette(img_path)

    show_image_cv2(silhouette_path, "Who's That Pokemon?")
    guess = input("Your guess: ").strip().lower()

    print()
    if guess == pokemon_name:
        print("✅ Correct!")
        show_image_cv2(img_path, f"It's {pokemon_name.title()}!")
        catch_pokemon_with_webcam(pokemon_name)
    else:
        print(f"❌ Nope! It was {pokemon_name.title()}!")
        show_image_cv2(img_path, f"It's {pokemon_name.title()}!")

    # Show caught Pokemon
    if CAUGHT_LIST:
        print("\n📦 Your Pokedex:")
        for p in CAUGHT_LIST:
            print(f"- {p}")

    # Delete all images after game ends
    for file in os.listdir(IMG_DIR):
        file_path = os.path.join(IMG_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    print("\n🧹 Assets cleaned up!")


if __name__ == "__main__":
    main()