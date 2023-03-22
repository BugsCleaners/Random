# Import necessary modules
from sqlalchemy import create_engine # for managing database connections and executing SQL queries
from sqlalchemy import text
import asyncio # for writing asynchronous code that can run concurrently with other tasks
import multiprocessing # for running multiple processes in parallel
import cv2 # for computer vision tasks such as image processing and video capture
import base64 # for encoding and decoding data in base64 format
import json # for parsing and manipulating JSON data
from datetime import datetime as dt # for working with dates and times
import time # for measuring time durations
import face_recognition # for facial recognition tasks
import numpy as np # for numerical computations
import pandas as pd # for data manipulation and analysis
import random # for generating random numbers and selecting random elements from lists
import socketio # for building real-time web applications using websockets
import uvicorn # for running Python web servers that support asynchronous code
import asyncio # for writing asynchronous code that can run concurrently with other tasks
import json # for parsing and manipulating JSON data
import random # for generating random numbers and selecting random elements from lists
import bcrypt # for encrypting and decrypting passwords
import csv # for reading and writing CSV files
import warnings # for suppressing warnings that may be displayed during program execution

# Suppress warnings displayed during program execution
warnings.filterwarnings("ignore")

# Create a connection to a MySQL database running on localhost:3306 with the database name attendance_ws
engine = create_engine("mysql+pymysql://root:@localhost:3306/attendance_ws",)

# Define a function to receive messages from a Socket.IO server
def socket2_process(namespace_dict, queue_dict):

    # Create an AsyncServer instance with CORS allowed from any origin and using the ASGI async mode
    sio = socketio.AsyncServer(cors_allowed_origins= '*', async_mode='asgi')
    
    # Create an ASGI app using the Socket.IO server instance
    app = socketio.ASGIApp(sio)

    # Define event handlers for the 'connect', 'disconnect', and 'website_functions' events
    @sio.event
    async def connect(sid, environ):
        print('connected', sid)

    @sio.event
    async def disconnect(sid):
        print('disconnected', sid)

    # Define an event handler for the 'login_credentials' event
    @sio.event
    async def login_credentials(sid, message):

        # Define a dictionary of camera IDs and their associated RTSP URLs
        cam_dict = {'camera_1':'rtsp://admin:Admin@54321@192.168.1.222/Streaming/Channels/101',
                    'camera_2':'rtsp://admin:CMBCSU@192.168.1.62/Streaming/Channels/101'
}  

        print(f"Received following message: {message}")
        JSON_dict = json.loads(message)           

        # If the 'password' key is in the message, check the login credentials
        if 'password' in JSON_dict.keys(): 
            
            # Query the database for the username and password
            user_query = engine.execute(text(f"SELECT username FROM developer WHERE username = '{JSON_dict['username']}';"))
            pass_query = engine.execute(text(f"SELECT password FROM developer WHERE username = '{JSON_dict['username']}';"))
            username = user_query.fetchall()[0][0]
            password = pass_query.fetchall()[0][0]
            
            # If the entered username and password combination match the hashed user and pass in the database, allow entry. 
            if username == JSON_dict['username'].lower() and bcrypt.checkpw(JSON_dict['password'].encode('utf-8'), password.encode('utf-8')):
                
                # If the device ID is 0, assign a new camera to the device and update the camera parameters
                if JSON_dict['device_id'] == 0:
                    temp_dict = {}
                    temp_dict['device_id'] = engine.execute(text(f"SELECT device_id FROM devices WHERE occ_flag = 0;")).fetchall()[0][0]
                    engine.execute(text(f"UPDATE `devices` SET `occ_flag`='1' WHERE device_id = '{temp_dict['device_id']}'"))
                    camera_temp_db = engine.execute(text(f"SELECT camera_id FROM devices WHERE device_id = '{temp_dict['device_id']}';")).fetchall()[0][0]
                    namespace_dict[camera_temp_db]['camera'] = cam_dict[camera_temp_db] 
                    temp_dict['response'] = 'first_login'
                    temp_dict['camera_id'] = camera_temp_db
                    json_string = json.dumps(temp_dict)
                    await sio.emit('login_response', json_string)
                    await asyncio.sleep(0.05)
                
                # If the device ID is not 0, update the camera parameters for the specified camera                
                else:  
                    temp_dict = {}
                    temp_dict['response'] = 'login_successful'
                    camera_temp = JSON_dict['camera_id']
                    namespace_dict[camera_temp]['camera'] = cam_dict[camera_temp]
                    temp_dict['camera_id'] = camera_temp
                    json_string = json.dumps(temp_dict)
                    await sio.emit('login_response', json_string)          
                    await asyncio.sleep(0.05) 

            else:
                print('user NOT authenticated')

    @sio.event
    async def website_functions(sid, message):
        
        # Initialize variables for capture and reset parameters and parse the JSON message received
        capture_param = ''
        reset_param = ''
        JSON_dict = json.loads(message)

        print(f"Received following message: {message}")

        # Retrieve the camera ID from the JSON message
        camera_temp = JSON_dict['camera_id']

        # Handle key presses from the website
        if 'key_pressed' in JSON_dict.keys():

            # If the 'id_attendance_pressed' key was pressed, pause detection
            if 'id_attendance_pressed' in JSON_dict.values():
                print('pause detection')
                namespace_dict[camera_temp]['status'] = 2

            # If the 'not_me_pressed' key was pressed, pause detection and record an "notme" attendance entry
            elif 'not_me_pressed' in JSON_dict.values():
                print('pause detection')
                namespace_dict[camera_temp]['status'] = 2
                
                # Insert attendance record into the database
                engine.execute(text(f"INSERT INTO `attendance`(`employee_id`, `type`, `camera_id`, `date_time`) VALUES ('{str(namespace_dict[camera_temp]['entry'])}','{'notme_'+namespace_dict[camera_temp]['detection_type']}','{camera_temp}','{dt.now()}');"))
                
                # Write attendance record to a CSV file
                with open('/home/admin-ai/Desktop/WS_FR_Project_auth/CSVs/attendance.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f'{str(namespace_dict[camera_temp]["entry"])}', f'notme_{namespace_dict[camera_temp]["detection_type"]}', f'{camera_temp}', f'{dt.now()}'])

            # If the 'confirm_dynamic_attendance' key was pressed, record an attendance entry            
            elif 'confirm_dynamic_attendance' in JSON_dict.values():
                capture_param = '1'
                
                # Insert attendance record into the database
                engine.execute(text(f"INSERT INTO `attendance`(`employee_id`, `type`, `camera_id`, `date_time`) VALUES ('{str(namespace_dict[camera_temp]['entry'])}','{namespace_dict[camera_temp]['detection_type']}','{camera_temp}','{dt.now()}');"))
                
                # Write attendance record to a CSV file
                with open('/home/admin-ai/Desktop/WS_FR_Project_auth/CSVs/attendance.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f'{str(namespace_dict[camera_temp]["entry"])}', f'{namespace_dict[camera_temp]["detection_type"]}', f'{camera_temp}', f'{dt.now()}'])

                reset_param = '1'
                namespace_dict[camera_temp]['sleep'] = 2

            # If the 'cancel_dyn_att_pressed' or 'cancel_id_attend_pressed' key was pressed, reset the camera parameters
            elif 'cancel_dyn_att_pressed' or 'cancel_id_attend_pressed' in JSON_dict.values():
                namespace_dict[camera_temp]['sleep'] = 0.75
                reset_param = '1'

        # If the 'submit_ID_attendance_pressed' key was pressed, validate the ID and update the camera parameters
        if 'submit_ID_attendance_pressed' in JSON_dict.keys():
            id_input = JSON_dict['submit_ID_attendance_pressed']
            temp_dict = {}
            if id_input != '' and id_input.isdigit() and len(id_input) <= 5:
                
                # Check if the ID is valid by querying the database
                query = engine.execute(text(f"SELECT name FROM employee_ids WHERE employee_id = {id_input};"))
                if len(query.fetchall()) != 0:
                    namespace_dict[camera_temp]['detection_type'] = 'id_attendance'
                    namespace_dict[camera_temp]['entry'] = id_input
                    namespace_dict[camera_temp]['status'] = 1
                
                # If the ID is not in DB send an error response
                else:
                    temp_dict['response'] = '3'
                    temp_dict['camera_id'] = camera_temp
                    json_string = json.dumps(temp_dict)
                    await sio.emit('error_response', json_string)
                    await asyncio.sleep(0.01) 
                    
            # If the ID is not valid send an invalidity reponse
            else: 
                temp_dict['response'] = '4'
                temp_dict['camera_id'] = camera_temp
                json_string = json.dumps(temp_dict)
                await sio.emit('error_response', json_string)
                await asyncio.sleep(0.01) 
            
            await asyncio.sleep(0.05) 

        # If capture_param is '1' and the detection type is 'id_attendance', capture the frame and save it to a file
        if capture_param == '1' and namespace_dict[camera_temp]['detection_type'] == 'id_attendance':
            temp_frame = queue_dict[camera_temp].get()
            capture_param = '0'
            p="/home/admin-ai/Desktop/WS_FR_Project_auth/id_captures/"+str(camera_temp)+'_'+str(namespace_dict[camera_temp]['entry'])+dt.now().strftime("_%d-%m_%H-%M-%S")+".png"
            cv2.imwrite(p, temp_frame)

        # If reset_param is '1', reset the camera parameters
        if reset_param == '1':
            namespace_dict[camera_temp]['entry'] = ''
            namespace_dict[camera_temp]['status'] = 0
            namespace_dict[camera_temp]['detection_type'] = ''
            reset_param = '0'

    # Start the server using uvicorn
    uvicorn.run(app, host="192.168.1.158", port=8854, ssl_certfile = '/home/admin-ai/Desktop/WS_FR_Project_auth/nginx/nginx-certificate.crt', ssl_keyfile = '/home/admin-ai/Desktop/WS_FR_Project_auth/nginx/nginx.key')

# This function handles the sending of detection responses (When a face is detected or ID is written) 
def socket1_process(namespace_dict):

    # Read quotes from a CSV file
    quote_df = pd.read_csv('/home/admin-ai/Desktop/WS_FR_Project_auth/CSVs/quotes.csv')

    # Create an asynchronous SocketIO server with CORS allowed origins
    sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
    app = socketio.ASGIApp(sio)

    # Define a SocketIO event handler that handles connection events
    @sio.on('connect')
    async def connect(sid, environ):
        print('Socket1 Connected!')   

    # Define a SocketIO event handler that sends detection responses to clients
    async def socket1_handler(sid, camera_id):
        while True:
            # Filter the namespace_dict to find cameras with detected entries and status of 1
            camera_detect = [key for key, value in namespace_dict.items() if value['entry'] != '' and value['status'] == 1 and key == camera_id]

            # If there is a detected entry for this camera and the client's unique ID matches, send a detection response to the client            
            if len(camera_detect) > 0 and namespace_dict[camera_id]['unique_id'] == sid: 
                for camera_temp_detect in camera_detect:    
                    temp_dict = {}
                    
                    # Set the response type to 2 for ID attendance, and 1 for face recognition                    
                    if namespace_dict[camera_temp_detect]['detection_type'] == 'id_attendance':
                        temp_dict['response'] = '2'                                       
                    else:
                        temp_dict['response'] = '1'
                        namespace_dict[camera_temp_detect]['detection_type'] = 'face_recognition'
                    temp_dict['id'] =  str(namespace_dict[camera_temp_detect]['entry'])     
                    temp_name = engine.execute(text(f"SELECT name FROM employee_ids WHERE employee_id = {str(namespace_dict[camera_temp_detect]['entry'])};"))

                    # Get the name associated with the employee ID 
                    temp_dict['name'] = temp_name.fetchall()[0][0]
                    
                    # Get a random quote from the CSV file                    
                    temp_dict['quote'] = quote_df.loc[random.randint(0, len(quote_df)-1),'quote']
                    temp_dict['camera_id'] = camera_temp_detect
                    json_string = json.dumps(temp_dict)
                    
                    # Set the status to 2 to indicate that the entry has been processed                    
                    namespace_dict[camera_temp_detect]['status'] = 2     
                    
                    # Send the detection response as a JSON string to the client                    
                    await sio.emit('detection_response', json_string, room=sid)   
                    await asyncio.sleep(0.01) 
            await asyncio.sleep(0.05) 


    # Define a SocketIO event handler that handles sharing of camera IDs
    @sio.event
    async def camera_share_id(sid, message):
        JSON_dict_rcv = json.loads(message) 
        print('JSON_dict_rcv', JSON_dict_rcv)
        camera_ = JSON_dict_rcv['camera_id']
        
        # Add the client's unique ID to the namespace_dict for the specified camera
        namespace_dict[camera_]['unique_id'] = sid
        print(namespace_dict[camera_]['unique_id'] + 'for ' + camera_)
        
        # Start a background task to handle sending of detection responses for the specified camera        
        sio.start_background_task(socket1_handler, sid, camera_)

    # Run the SocketIO server with the specified configuration
    uvicorn.run(app, host="192.168.1.158", port=8853, ssl_certfile = '/home/admin-ai/Desktop/WS_FR_Project_auth/nginx/nginx-certificate.crt', ssl_keyfile = '/home/admin-ai/Desktop/WS_FR_Project_auth/nginx/nginx.key')

# Defining a function called "socket3_process" that is responsible for sending video feeds to clients
def socket3_process(namespace_dict):

# Creating a new Socket.IO async server object with 'asgi' mode and allowing cross-origin requests
    sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
    
    # Creating a new ASGI application using the above server object    
    app = socketio.ASGIApp(sio)

    # Defining a new async function called "socket2_handler_video" that takes a single input parameter called "sid"
    async def socket3(sid):
        while True:
            
            # Creating a new dictionary called "feed_dict" using a dictionary comprehension. 
            # It contains base64-encoded image frames from the cameras which are not empty and have a non-None frame value.            
            feed_dict = {key: base64.b64encode(value['frame'][0].tobytes()).decode('ascii') for key, value in namespace_dict.items() if value['camera'] != '' and value['frame'][0] is not None }
                        
             # If "feed_dict" is not empty
            if feed_dict:

                json_string = json.dumps(feed_dict)
                await sio.emit('video_feed',json_string)

            # Pausing the execution of this coroutine for 0.01 seconds
            await asyncio.sleep(0.01)
    
    # Defining a new Socket.IO event handler function called "connect" which takes two input parameters called "sid" and "environ"
    @sio.on('connect')
    async def connect(sid, environ):
        print("Connected ", sid)

        # Starting a new background task to run the "socket2_handler_video" coroutine in the server
        sio.start_background_task(socket3, sid) 

    # Running the ASGI application using Uvicorn with the specified host, port, and SSL certificate/key files
    uvicorn.run(app, host="192.168.1.158", port=8855, ssl_certfile = '/home/admin-ai/Desktop/WS_FR_Project_auth/nginx/nginx-certificate.crt', ssl_keyfile = '/home/admin-ai/Desktop/WS_FR_Project_auth/nginx/nginx.key')

# Defining a function called "ThreadedCamera" that takes three input parameters called "shared_variable", "queue", and "key"
def ThreadedCamera(shared_variable, queue, key):

    print(f'Threaded Camera Process for {key} has started!')

    # Setting some initial values for some variables
    FPS = 1/100
    flag_camera = 1
    flag_open = 0
    counter = 0
    while True:

        # Incrementing the counter variable
        counter = counter + 1

        # If the camera source path is not empty and the "flag_camera" variable is 1
        if shared_variable[key]['camera'] != '' and flag_camera == 1:
            print(f'shared_variable[key]["camera"] for {key}', shared_variable[key]['camera'])
            
            # Getting the camera source path as a string            
            src = str(shared_variable[key]['camera'])

            # Creating a new VideoCapture object using the camera source path
            capture = cv2.VideoCapture(src)
            
            # Setting the VideoCapture object's buffer size to 1            
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
            
            # Setting the "flag_camera" variable to 0 to indicate that the camera has been initialized            
            flag_camera = 0    
            
            # Setting the "flag_open" variable to 1 to indicate that the camera is open            
            flag_open = 1            

        # If the camera is open
        if flag_open == 1:

            # Reading a frame from the camera                        
            (status, frame) = capture.read()

            # If the frame was successfully read
            if status:
                try:
                    # Adding the frame to the queue for the specified camera
                    queue[key].put_nowait(frame)

                    # If this is camera 2 and "cam2" variable is 1
                    if key == 'camera_2':
                        # Pausing the thread for 0.01 seconds to initiate the camera
                        time.sleep(0.01)
                
                # If there is no space left in the queue for the specified camera                
                except:

                    # Removing the oldest frame from the queue for the specified camera
                    queue[key].get() 
                
                # Encoding the frame as a JPEG image with quality 80 and storing it in the shared dictionary for the specified camera                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                shared_variable[key]['frame'][0] = cv2.imencode('.jpg', frame, encode_param)[1]

            # If the frame could not be read from the camera
            else:
                print(f'camera {key} is broken')
        
        # Pausing the thread for "FPS" seconds
        time.sleep(FPS)


#Face Recognition
def face_recognition_live(shared_variable, queue, key):

    print(f'Face Recognition Process for {key} has started!')
    # Defining variables, initialize some variables
    previous_entry = ""
    entry = ""
    entry_counter = 1
    process_this_frame = 2
    max_area = 0

    # convert encondings to list of arrays for recognition
    df_encodings = pd.read_csv('/home/admin-ai/Desktop/WS_FR_Project_auth/CSVs/encodings.csv', header=None)
    encondings_list = df_encodings.iloc[:,1:].values.tolist()
    known_face_encodings = [np.asarray(l) for l in encondings_list]
    known_face_names= list(df_encodings.iloc[:,0]).copy()

    # Initialize some variables
    # Create arrays of known face encodings and their names
    face_locations = []
    face_encodings = []
    face_names = []
    
    # Continuously run loop
    while True: 

        # Get image in que
        img__ = queue[key].get()

        # Sleep detection 
        if shared_variable[key]['sleep'] != 0:
            time.sleep(shared_variable[key]['sleep'])
            shared_variable[key]['sleep'] = 0

        # Process frame for face detection
        if process_this_frame % 2 == 0 and str(shared_variable[key]['status']) == '0':

            #enter only if there's an image (frame) in que
            if type(img__) is np.ndarray:

                # Resize frame of video to 1/4 size for faster face recognition processing            
                small_frame = cv2.resize(img__, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)

                # Checking if any faces were detected in the current frame
                if len(face_locations) != 0:
                    
                    
                    for i, value in enumerate(face_locations):
                        
                        # Calculating the area of the current face bounding box
                        area = (value[2]-value[0])*(value[1]-value[3])

                        # Updating the maximum area if the current area is larger
                        if area>max_area:
                            max_area=area
                            face_locations_max = [value]

                    # Checking if the maximum area of the detected faces is large enough to be considered a valid face
                    if max_area >= 2500:
                        print(f'Area of detected face in {key}: ',max_area)                                                            

                        # Encoding the detected face as a vector                                
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations_max)                
                        face_names = []

                        # Looping through each encoded face vector
                        for face_encoding in face_encodings:

                            # Comparing the current face vector to the known face vectors to see if there is a match
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = 0.55)

                            # Finding the index of the known face that is the best match for the current face
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)

                            # Setting some variables to keep track of the detected face
                            previous_entry = entry
                                                    
                            if matches[best_match_index]:

                                entry = known_face_names[best_match_index]

                                # If the same face was detected multiple times in a row
                                if previous_entry == entry:
                                    entry_counter = entry_counter + 1
                                else:
                                    entry_counter = 1
                                
                                # If the same face has been detected three times in a row
                                if entry_counter == 3:
                                    print(f'Detected face 3 times in {key}')

                                    # Storing the detected face's name and status in the shared dictionary for the current camera
                                    shared_variable[key]['entry'] = entry
                                    shared_variable[key]['status'] = 1
                                    
                                    # Pausing the thread for 0.05 seconds
                                    time.sleep(0.05)                          

                                    # Resetting some variables to their initial values
                                    print(f'Resetting face detection variables for {key}')
                                    previous_entry = ""
                                    entry = ""
                                    entry_counter = 1
                                    process_this_frame = 0
                                    max_area = 0                            
                            face_names.append(entry)            
            # If there is no frame
            else:
                print(f'Image not available in {key}')    
        # Adding 1 to the frame counter so that it will processes frames based on a set calculation            
        process_this_frame = process_this_frame + 1

# Defining the main function for running multiprocesses
def main():  

    # Creating a new SQLAlchemy engine and executing a SQL query to get all camera IDs from the "devices" table 
    camera_query = engine.execute(text(f"SELECT camera_id FROM devices;")).fetchall()
    
    # Extracting the camera IDs from the query result
    camera_names = [x[0] for x in camera_query]

    # Creating a new multiprocessing Manager object
    manager = multiprocessing.Manager()

    # Creating a dictionary called "queue_dict" with one multiprocessing queue object for each camera ID
    queue_dict = {f'{name}': multiprocessing.Queue(1) for name in camera_names}
    
    # Creating a dictionary called "namespace_dict" with one multiprocessing dictionary object for each camera ID
    namespace_dict = {f'{name}': manager.dict() for name in camera_names}
    
    # Creating a list called "frame_lists" with one multiprocessing list object for each camera ID
    frame_lists = [manager.list([None]) for camera in camera_names]

    # Updating the "namespace_dict" dictionary for each camera ID with some initial values
    for index, camera in enumerate(camera_names):
        namespace_dict[camera] = manager.dict({'frame': frame_lists[index], 'entry':'', 'status':0, 'sleep':0,
                                                'detection_type':'', 'camera':'', 'unique_id':''})                                                    

    # Creating new multiprocessing processes for each of the required tasks
    socket1_multiprocess = multiprocessing.Process(target=socket1_process, args=(namespace_dict,))
    socket2_multiprocess = multiprocessing.Process(target=socket2_process, args=(namespace_dict, queue_dict,))
    socket3_multiprocess = multiprocessing.Process(target=socket3_process, args=(namespace_dict,))
    camera_processes_dict = {key: multiprocessing.Process(target=ThreadedCamera, args=(namespace_dict, queue_dict, key,)) for key in queue_dict}
    fr_processes_dict = {key: multiprocessing.Process(target=face_recognition_live, args=(namespace_dict, queue_dict, key)) for key in queue_dict}
    
    # Starting the multiprocessing processes
    socket1_multiprocess.start() 
    socket2_multiprocess.start() 
    socket3_multiprocess.start() 

    for key in camera_processes_dict.keys():        
        camera_processes_dict[key].start()
        fr_processes_dict[key].start()

    # Running an infinite loop to keep the program running
    while True:
        pass

# Running the "main" function if the script is being run as the main program
if __name__ == '__main__':
        main()
        