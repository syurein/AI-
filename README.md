
**<ソースコード>**

１．
from google.colab import drive
drive.mount('/content/drive')
この部分では、Google driveに接続します


２．
!pip install uvicorn
!pip install fastapi
!pip install python-multipart
!pip install simple_lama_inpainting
!pip install diffusers
!pip install python-multipart opencv-python-headless pillow


このような部分ではライブラリをインストールしています



３．
import os


# DeepFillv2 セットアップ
HOME = "/content"
%cd {HOME}
!git clone https://github.com/vrindaprabhu/deepfillv2_colab.git
!gdown "https://drive.google.com/u/0/uc?id=1uMghKl883-9hDLhSiI8lRbHCzCmmRwV-&export=download"
!mv /content/deepfillv2_WGAN_G_epoch40_batchsize4.pth deepfillv2_colab/model/deepfillv2_WGAN.pth


# GroundingDINO セットアップ
%cd {HOME}
!git clone https://github.com/IDEA-Research/GroundingDINO.git
%cd {HOME}/GroundingDINO
!pip install -q -e .
!pip install -q roboflow


# GroundingDINOのコンフィグとウェイトのダウンロード
CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))


!mkdir -p {HOME}/weights
%cd {HOME}/weights
!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
%cd {HOME}/GroundingDINO

ここではそれぞれのライブラリをインストールして、環境設定しています。
一番最初にgithubからdeepfillv2をダウンロードして、その後grounding DINOをダウンロードして、最後にディレクトリを移動して終了です。


＜それぞれの関数の説明＞

def quantize_model(model):
    # モデルを動的量子化
    model_quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model_quantized
①CPU実行時に少しでも軽量化するための量子化コード
モデルを与えると、量子化したモデルを返す。



def decide_to_object(risk_level):
    tex = [
        'text','Name tag', 'License plate', 'Mail', 'Documents', 'QR codes',
        'barcodes', 'Map', 'Digital screens', 'information board',
        'signboard', 'poster', 'sign', 'logo', 'card', 'window', 'mirror',
        'Famous landmark', 'cardboard', 'manhole', 'utility pole'


    ]
    #この配列の要素の順番を変えると消える順番が変わる。
    risk_level = int(risk_level / 10)*(len(tex)/10)#個数決定
    return tex[:int(risk_level)+1]


②リスクレベルに応じて、消す要素の個数を変更する関数
仕組みとしては、risk_level/100にして、まずは全体の何パーセントかを求め、その後、全体のオブジェクトの個数にパーセンテージをかけることで個数を出している。


def create_mask(image, x1, y1, x2, y2):
    # Create a black image with the same size as the input image
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)


    # Draw a white rectangle on the mask where the object is located
    cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)


    return mask

③この関数に座標を二地点と、imageを与えると、マスク画像を返す関数
numpyとopenCVの機能で実装している


#この下のコードは特定の領域をマスクしないタイプのコード
def special_process_image(risk_level, image_path, point1, point2, thresholds=None):
    if thresholds is None:
        thresholds = {}
    detection_model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    IMAGE_PATH = image_path
    LABELS = decide_to_object(risk_level)
    image_source, image = load_image(IMAGE_PATH)


    all_boxes = []
    all_logits = []


    for label in LABELS:
        boxes, logits, _ = predict(
            model=detection_model,
            image=image,
            caption=label,
            box_threshold=0.3,
            text_threshold=0.3
        )
        all_boxes.extend(boxes)
        all_logits.extend(logits)


    # Convert PyTorch tensor to NumPy array and ensure it's uint8
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)  # Ensure values are in [0, 255]


    # Initialize mask as black
    mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)


    for i, box in enumerate(all_boxes):
        x, y, w, h = box
        confidence = all_logits[i]  # Get the confidence score
        object_type = LABELS[i % len(LABELS)]  # Determine the object type


        # Use the threshold specific to the object type, default to 0.5 if not provided
        threshold = thresholds.get(object_type, 0.5)


        if confidence >= threshold:
            x1 = int((x - w / 2) * image_np.shape[1])
            y1 = int((y - h / 2) * image_np.shape[0])
            x2 = int((x + w / 2) * image_np.shape[1])
            y2 = int((y + h / 2) * image_np.shape[0])


            # Ensure coordinates are within the image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)


            mask = create_mask(image_np, x1, y1, x2, y2) | mask


    # Convert the two points into proper coordinates
    p1_x, p1_y = point1
    p2_x, p2_y = point2


    # Ensure points are within the image dimensions
    p1_x, p1_y = max(0, min(p1_x, image_np.shape[1])), max(0, min(p1_y, image_np.shape[0]))
    p2_x, p2_y = max(0, min(p2_x, image_np.shape[1])), max(0, min(p2_y, image_np.shape[0]))


    # Make the region defined by point1 and point2 black (don't mask this area)
    x_min = min(p1_x, p2_x)
    x_max = max(p1_x, p2_x)
    y_min = min(p1_y, p2_y)
    y_max = max(p1_y, p2_y)


    # Set the area defined by the rectangle to 0 (black)
    mask[y_min:y_max, x_min:x_max] = 0


    # Save the mask as an image
    final_image_pil = Image.fromarray(mask)
    final_image_pil.save("/content/final_mask.jpg")


    return "/content/final_mask.jpg"
⑤このコードに、画像のパスと、リスクレベルと、二地点の座標を与えると、その二地点で囲まれた場所以外に対して画像認識を行って、マスク画像を生成する。（特別バージョン）






def process_image(risk_level, image_path, thresholds=None):
    if thresholds is None:
        thresholds = {}
    detection_model = load_model(CONFIG_PATH, WEIGHTS_PATH)


    risk_level=int(risk_level)
    IMAGE_PATH = image_path
    LABELS = decide_to_object(risk_level)
    image_source, image = load_image(IMAGE_PATH)


    all_boxes = []
    all_logits = []


    for label in LABELS:
        boxes, logits, _ = predict(
            model=detection_model,
            image=image,
            caption=label,
            box_threshold=0.3,
            text_threshold=0.3
        )
        all_boxes.extend(boxes)
        all_logits.extend(logits)


    # Convert PyTorch tensor to NumPy array and ensure it's uint8
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)  # Ensure values are in [0, 255]


    # Initialize mask as black
    mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)


    for i, box in enumerate(all_boxes):
        x, y, w, h = box
        confidence = all_logits[i]  # Get the confidence score
        object_type = LABELS[i % len(LABELS)]  # Determine the object type


        # Use the threshold specific to the object type, default to 0.5 if not provided
        threshold = thresholds.get(object_type, 0.5)


        if confidence >= threshold:
            # Adjust the box coordinates, convert to integers
            x1 = int((x - w / 2) * image_np.shape[1])
            y1 = int((y - h / 2) * image_np.shape[0])
            x2 = int((x + w / 2) * image_np.shape[1])
            y2 = int((y + h / 2) * image_np.shape[0])


            # Ensure coordinates are within the image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)


            mask = create_mask(image_np, x1, y1, x2, y2) | mask


    # Save the mask as an image
    final_image_pil = Image.fromarray(mask)
    final_image_pil.save("/content/final_mask.jpg")


    return "/content/final_mask.jpg"


⑥このコードは、通常使用バージョン。画像のパスと、リスクレベルとthresholderを与えると画像認識を行える。thresholderをいじることで、画像認識の精度をいじれる。
thresholds = {
    'text': 0.4,
    'Name tag': 0.2,
    'License plate': 0.4,
    'Mail': 0.3,
    'Documents': 0.5,
    'QR codes': 0.4,
    'barcodes': 0.4,
    'Map': 0.5,
    'Digital screens': 0.6,
    'information board': 0.5,
    'signboard': 0.3,
    'poster': 0.8,
    'sign': 0.3,
    'logo': 0.3,
    'card': 0.4,
    'window': 0.2,
    'mirror': 0.2,
    'Famous landmark': 0.7,
    'cardboard': 0.6,
    'manhole': 0.6,
    'utility pole': 0.7
}

こんなように設定することができる。


最終的にマスク画像のパスを返す。中にGroundingDINOやYOLOの処理が含まれていて（関数ではなく）、中で、マスク作成の関数を呼び出している。




def inpaint_image_with_mask(image_path, mask_path, output_path, inpaint_radius=5, inpaint_method=cv2.INPAINT_TELEA):
    """
    マスク画像を使用して元画像のインペイントを行う関数。


    Parameters:
    - image_path: 元画像のパス
    - mask_path: マスク画像のパス（修復したい領域が白、その他が黒）
    - output_path: インペイント結果の出力パス
    - inpaint_radius: インペイントの半径（デフォルトは5）
    - inpaint_method: インペイントのアルゴリズム（デフォルトはcv2.INPAINT_TELEA）


    Returns:
    - inpainted_image: インペイントされた画像
    """
    # 画像とマスクを読み込み
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # マスクはグレースケールで読み込み


    # マスク画像が正常に読み込めたかチェック
    if image is None:
        raise ValueError(f"元画像が見つかりません: {image_path}")
    if mask is None:
        raise ValueError(f"マスク画像が見つかりません: {mask_path}")


    # マスク画像が元画像と同じサイズでない場合、リサイズ
    if image.shape[:2] != mask.shape[:2]:
        print(f"マスク画像のサイズを元画像に合わせてリサイズします: {mask.shape} -> {image.shape[:2]}")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))


    # インペイント処理
    inpainted_image = cv2.inpaint(image, mask, inpaint_radius, inpaint_method)


    # インペイント結果を保存
    cv2.imwrite(output_path, inpainted_image)


    return inpainted_image



⑦この関数では、openCVのインペイント機能を呼び出して処理している。openCVはリサイズして、マスク画像と元画像のサイズを同じにしなくてはいけないのでそういう処理を含めている。関数で呼び出しているわけではない。最終的にインペイント結果を返す（パスではない）



def inpaint_image_with_mask1(img_path, mask_path, output_path, resize_factor=0.5):
    print('lama')
    # 画像とマスクを読み込み
    image = Image.open(img_path)
    mask = Image.open(mask_path).convert('L')  # マスクをグレースケールに変換


    # 画像とマスクのサイズを合わせる
    mask = mask.resize(image.size, Image.NEAREST)


    # SimpleLama インスタンスを作成
    simple_lama = SimpleLama()


    # インペイントの実行
    result = simple_lama(image, mask)


    # 出力画像をリサイズ
    new_size = (int(result.width * resize_factor), int(result.height * resize_factor))
    result = result.resize(new_size, Image.ANTIALIAS)


    # 結果を保存
    result.save(output_path)
    print(f"Inpainted image saved at {output_path}")


⑧この関数では、simple lamaの機能を呼び出して、それを処理している。最終的には、パスを変えさせる。詳しい処理の内容はコメントで書かれている通りだが、グレースケールに変換しないとうまく処理できないので含めている。また、リサイズも行っている。


def load_inpainting_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    return pipe


def inpaint_images(image_paths, mask_paths, output_dir="output_images"):
    # パイプラインの読み込み
    pipe = load_inpainting_pipeline()


    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # 画像とマスクを読み込み
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


        # 画像とマスクの読み込みチェック
        if image is None:
            raise ValueError(f"{image_path} が正しく読み込まれていません。ファイルパスを確認してください。")
        if mask is None:
            raise ValueError(f"{mask_path} が正しく読み込まれていません。ファイルパスを確認してください。")


        # マスクと画像のサイズ確認
        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(f"マスク {mask_path} のサイズが画像 {image_path} のサイズと一致しません。")


        # 画像とマスクをPIL形式に変換
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)


        # インペインティング処理の実行
        result = pipe(
            prompt="Fill the masked area with contextually appropriate background or scenery that matches the surrounding area.",
            negative_prompt="Exclude people, human figures, arms, and any human body parts from the generated content.",#生成したくないものを生成しないようにするプロンプト
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=100,        # ステップ数を増加
            guidance_scale=7.5             # 高いガイダンススケールを設定
        ).images[0]


        # 結果のリサイズと保存
        result_image = result.resize((image.shape[1], image.shape[0]), Image.LANCZOS)
        output_image_path = os.path.join(output_dir, f"output_{i+1}.png")
        result_image.save(output_image_path)


        # 処理が完了したことを通知
        print(f"{image_path} と {mask_path} のペアの処理が完了しました。結果は {output_image_path} に保存されました。")




⑨この関数では、stable diffusionの設定と実行を行える。imagepathとmaskpathを与えると勝手に処理をしてくれる。内部の処理としては、まずはstablediffusionのモデルパイプラインを読み込み、画像が本当にあるのかを確認してから、マスク画像と元画像をPIL画像形式に変換してからstablediffusionに投げることで処理を実行している。なお、pipe以降の部分でネガティブプロンプトとプロンプトを操作することで生成される画像が適切な物になるように処理している。
また、ステップ数を操作することで画像の品質をアップダウンさせたり、速度をアップダウンさせたりできる。50～100程度にしてください。

app = FastAPI()
# CORSミドルウェアの追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ここを適切なオリジンに設定することもできます
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


クロスオリジンを設定することで、APIへのアクセスを許可している。


def save_image(file, filename):
    """画像ファイルを指定ディレクトリに保存"""
    filepath = SAVE_DIR / filename
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file, buffer)
    return filepath


この関数に画像を与えると、画像を保存してくれる。最後にパスを返す。


@app.post("/create-mask-and-inpaint-opencv")
async def create_mask_and_inpaint_opencv(image: UploadFile = File(...), risk_level: int = Form(...)):
    input_path = save_image(image.file, "input.jpg")
    mask_path = process_image(risk_level, input_path, thresholds=thresholds)


    output_path = SAVE_DIR / "output_opencv.jpg"
    # OpenCVでインペイント
    inpaint_image_with_mask(input_path, mask_path, output_path)


    return FileResponse(output_path)
⑩このエンドポイントでは、openCVでの処理を行わせている。


#下のendpointは特定領域をマスクしないタイプのもの
@app.post("/create-mask-and-inpaint-simple-lama-special")
async def create_mask_and_inpaint_simple_lama(
    image: UploadFile = File(...),
    risk_level: int = Form(...),
    point1: tuple[int, int] = Form(...),  # フォームから座標を取得
    point2: tuple[int, int] = Form(...)
):
    # 入力画像を保存
    input_path = save_image(image.file, "input.jpg")
    mask_path = "/content/final_mask.jpg"  # マスク画像の保存パス
    output_path = "/content/output_simple_lama.jpg"




    # マスク画像を作成 (新しい関数を使用)
    process_image(risk_level, input_path, point1, point2, thresholds=thresholds)


    # SimpleLamaでインペイント
    inpaint_image_with_mask1(input_path, mask_path, output_path, resize_factor=1)


    return FileResponse(output_path)


⑪これも同じ。

def copy_image_to_folder(image_path, save_as_input):
    """
    Copies the given image to the appropriate folder based on the save_as_input flag.


    Parameters:
    image_path (str): The path of the image to be copied.
    save_as_input (int): 0 to save as input image, 1 to save as mask image.
    """
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist!")
        raise StopExecution


    # Set the destination folder and file name based on the save_as_input flag
    if save_as_input == 0:
        destination_path = "/content/deepfillv2_colab/input/input_img.png"
        print(f"Saving image as input: {destination_path}")
    elif save_as_input == 1:
        destination_path = "/content/deepfillv2_colab/input/mask.png"
        print(f"Saving image as mask: {destination_path}")
    else:
        print("Error: Invalid save_as_input value. It must be 0 (input) or 1 (mask).")
        raise StopExecution


    # Copy the image to the destination
    shutil.copy(image_path, destination_path)
    print(f"Image {image_path} copied to {destination_path} successfully!")




⑫この関数は、画像のパスを与えて、オプションを与えると画像を保存してくれる。
なお、今までの関数と違ってdeepfillv2のための場所に保存する。
いじらないほうがいい。オプションで0か1を指定することで、マスク画像と元画像のどちらかを指定できる。


def run_python_script(script_path):
    try:
        # Pythonスクリプトを新しいプロセスとして実行
        result = subprocess.run(['python', script_path], check=True, text=True, capture_output=True)
        # 実行が成功した場合、標準出力を表示
        print("Script output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running script: {e}")
        print("Script stderr:", e.stderr)

⑬この関数で既存のpythonプログラムを動かすことができる。


@app.post("/create-mask-and-inpaint-deepfillv2")
# エンドポイントの実装
async def deepfillv2_inpaint_endpoint(
    image: UploadFile = File(...),
    risk_level: int = Form(...)
):
    input_path = save_image(image.file, "input.jpg")
    save_as_input=0
    copy_image_to_folder(input_path, save_as_input)
    input_path=process_image(risk_level,input_path, thresholds=thresholds)
    save_as_input=1#option
    copy_image_to_folder(input_path, save_as_input)
    os.chdir(f"{HOME}/deepfillv2_colab")#一度ディレクトリを変更
    run_python_script('/content/deepfillv2_colab/inpaint.py')#プロセス用のコードを呼び出す
    #!python inpaint.py
    os.chdir(f"{HOME}/GroundingDINO") #元に戻す
    output_image_path='/content/deepfillv2_colab/output/inpainted_img.png'
    return FileResponse(output_image_path)


⑭この関数では、特殊な処理をしている。そのままではdeepfillv2を動かすことはできないので一度マスクと元画像を一度保存しておく。また、ディレクトリを移動して、deepfillv2をインポートできるようにする。そのままではインポートできないからである。また、最後には、grounding DINOのディレクトリに移動して、画像認識を実行できるようにしている。


@app.post("/create-mask-and-inpaint-stable-diffusion")
async def create_mask_and_inpaint_stable_diffusion(image: UploadFile = File(...), risk_level: int = Form(...)):
    input_path = save_image(image.file, "input.jpg")
    mask_path = process_image(risk_level, input_path, thresholds=thresholds) # マスク画像を作成


    output_path = SAVE_DIR / "output_stable_diffusion.png"


    # 保存された画像の有効性を確認
    validate_image(input_path)




    # マスクと元画像のサイズが一致しない場合はリサイズ
    resize_mask_if_needed(mask_path, input_path)


    # Stable Diffusionでインペイント
    inpaint_images([input_path], [mask_path], output_dir=SAVE_DIR)


    return FileResponse(SAVE_DIR / "output_1.png")
⑮このコードは、シンプルラマと同じく画像に対してstablediffusionのインペイント機能を実行している。


# ベクトル化対象のオブジェクトリスト
TEXT_PROMPTS = [
     'text','Name tag', 'License plate', 'Mail', 'Documents', 'QR codes',
        'barcodes', 'Map', 'Digital screens', 'information board',
        'signboard', 'poster', 'sign', 'logo', 'card', 'window', 'mirror',
        'Famous landmark', 'cardboard', 'manhole', 'utility pole'
]
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3


# クラスタリング結果をJSONファイルから読み込む関数
def load_sums_from_json(filepath):
    with open(filepath, 'r') as json_file:
        sums = json.load(json_file)
    return sums


# ベクトルデータをJSONファイルから読み込む関数
def load_vectors_from_json(filepath):
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    return data


# 新しい画像を分類する関数
def classify_new_image(new_image_vector, sums_data, loaded_vectors, loaded_object_names, k=1):
    cluster_centers = []
    for cluster in sums_data:
        indices = [loaded_object_names.index(obj_name) for obj_name in cluster]
        cluster_vectors = np.array([loaded_vectors[obj_name] for obj_name in cluster])
        cluster_center = np.mean(cluster_vectors, axis=0)
        cluster_centers.append(cluster_center)


    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(cluster_centers, range(len(cluster_centers)))


    new_image_label = knn.predict([new_image_vector])
    return new_image_label[0]


# 画像ベクトル化の処理（例としての関数）
def process_image_vec(image_path):
    detection_model = load_model(CONFIG_PATH, WEIGHTS_PATH)


    image_source, image = load_image(image_path)
    object_vector = np.zeros(len(TEXT_PROMPTS))


    for i, text_prompt in enumerate(TEXT_PROMPTS):
        boxes, logits, phrases = predict(
            model=detection_model,
            image=image,
            caption=text_prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        if len(logits) > 0:
            logits_np = np.array(logits)
            object_vector[i] = np.sum(logits_np)
    print(object_vector)
    return object_vector.tolist()


⑮この関数の中では三つの処理を実行している。
まず、画像を配列の中に入っている要素でベクトル化する（符号化）
次に事前にクラスタリングされているJSONデータを読み込み、新規画像がその中のどれに近いのかを分析している。最終的にクラスタ番号を返す。



@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    image_path = "/tmp/temp_image.jpg"


    # アップロードされた画像を保存
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())


    # 画像をベクトル化
    new_image_vector = process_image_vec(image_path)


    # JSONファイルからデータを読み込む
    json_filepath = "/content/drive/MyDrive/lastsum/output_vectors.json"
    loaded_data = load_vectors_from_json(json_filepath)
    loaded_vectors = {obj_name: np.array(vector) for obj_name, vector in loaded_data.items()}
    loaded_object_names = list(loaded_vectors.keys())


    # 既存のクラスタリング結果を読み込む
    sums_data = load_sums_from_json("/content/drive/MyDrive/lastsum/sums_data.json")


    # 新しい画像がどのクラスタに分類されるかを判定
    new_image_cluster = classify_new_image(new_image_vector, sums_data, loaded_vectors, loaded_object_names)


    return {"danger":dangerarray[int(new_image_cluster - 1)]}




この関数では、先ほどの関数を呼び出しつつ、帰ってきたクラスタ番号が危険度何に属するのかを分析する。
dangerarray=[10,30,90,50,80,20,40,70,100,60]



一番上のこの配列をいじれば、危険度を設定しなおせる。

単純に配列に番号を与えてその要素を取り出すだけの仕組みである。


def run_fastapi():
    if __name__ == "__main__":
        nest_asyncio.apply()
        uvicorn.run(app, host="0.0.0.0", port=8000)


def run_ngrok():
    os.system("ngrok http --domain=wired-kitten-adequately.ngrok-free.app 8000")#固定URLを発行


# スレッドを作成
fastapi_thread = threading.Thread(target=run_fastapi)
ngrok_thread = threading.Thread(target=run_ngrok)


# 二つのスレッドを動かす
fastapi_thread.start()
ngrok_thread.start()
fastapi_thread.join()
ngrok_thread.join()


この部分では、ngrokとfastAPIが同時に動くように並列化の処理を行っている。thredsという技術を用いている。


#統合用コード,vector作成
import os
import json
import numpy as np


# ベクトル化対象のオブジェクトリスト
TEXT_PROMPTS = [
    'text', 'mirror', 'window', 'cardboard', 'poster', 'sign', 'manhole',
    'logo', 'signboard', 'utility pole','Name tag','License plate','card','Mail or envelope',
    'Digital screens','QR codes','barcodes','documents','information board','Map',
    'Famous landmark'
]
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3


def process_image_vec(image_path):
    image_source, image = load_image(image_path)
    object_vector = np.zeros(len(TEXT_PROMPTS))


    for i, text_prompt in enumerate(TEXT_PROMPTS):
        boxes, logits, phrases = predict(
            model=detection_model,
            image=image,
            caption=text_prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        if len(logits) > 0:
            logits_np = np.array(logits)
            object_vector[i] = np.sum(logits_np)
    print(object_vector)
    return object_vector.tolist()


def process_images_in_folder(folder_path, output_json_path):
    results = {}


    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            object_vector = process_image_vec(image_path)
            results[filename] = object_vector


    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)


# 画像が格納されているフォルダのパスと出力JSONファイルのパス
folder_path = "/content/drive/MyDrive/lastsum"
output_json_path = "/content/drive/MyDrive/lastsum/output_vectors.json"


# フォルダ内の画像を処理してJSONに保存
process_images_in_folder(folder_path, output_json_path)


このコードでgoogle driveのlastsumというところにある画像全部をベクトル化し、それらをoutput_vectors.jsonとして保存している。


import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans, Birch, SpectralClustering, MeanShift, OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def load_from_json(filepath):
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    return list(data.values()), list(data.keys())


# JSONファイルのパスを入力
json_filepath = "/content/drive/MyDrive/lastsum/output_vectors.json"  # JSONデータが保存されたファイルパス


# JSONファイルからデータを読み込む
loaded_vectors, loaded_object_names = load_from_json(json_filepath)
X = np.array(loaded_vectors)
num = len(loaded_vectors)


# クラスタリング手法のリスト
clustering_methods = [
    ('KMeans', KMeans),
    ('MiniBatchKMeans', MiniBatchKMeans),
    ('AgglomerativeClustering', AgglomerativeClustering),
    ('DBSCAN', DBSCAN),
    ('Birch', Birch),
    ('SpectralClustering', SpectralClustering),
    ('MeanShift', MeanShift),
    ('OPTICS', OPTICS)
]


# クラスタリング手法を選択
print("Select a clustering method:")
for i, (name, method) in enumerate(clustering_methods):
    print(f"{i}: {name}")
method_index = int(input("Enter the number of the clustering method: "))
method_name, ClusteringMethod = clustering_methods[method_index]


# クラスタ数を選択する範囲
k_range = range(8, min(100, num))
# シルエットスコアの変化量を使って最適なクラスタ数を見つける
silhouette_scores = []


for k in k_range:
    if method_name in ['DBSCAN', 'MeanShift', 'OPTICS']:
        clustering = ClusteringMethod()
        labels = clustering.fit_predict(X)
        k = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        clustering = ClusteringMethod(n_clusters=k)
        labels = clustering.fit_predict(X)


    if len(set(labels)) > 1:
        silhouette_scores.append(silhouette_score(X, labels))


# シルエットスコアの変化量を計算
silhouette_deltas = np.diff(silhouette_scores)


# 変化量が最大のインデックスを取得
max_delta_index = np.argmax(silhouette_deltas)


# 最適なクラスタ数
optimal_k = k_range[max_delta_index + 1]  # np.diffの結果は1つ少ないため、インデックスを調整


print(f'Optimal number of clusters based on silhouette score change: {optimal_k}')


# シルエットスコアと変化量のプロット
plt.figure()
plt.plot(k_range[:len(silhouette_scores)], silhouette_scores, marker='o', label="Silhouette Score")
plt.plot(k_range[1:len(silhouette_scores)], silhouette_deltas, marker='x', label="Silhouette Score Change", color='red')
plt.xlabel('Number of clusters')
plt.ylabel('Score / Change')
plt.title(f'Silhouette Scores and Changes for {method_name}')
plt.legend()
plt.show()


# 最適なクラスタ数でクラスタリング
if method_name in ['DBSCAN', 'MeanShift', 'OPTICS']:
    clustering = ClusteringMethod()
else:
    clustering = ClusteringMethod(n_clusters=optimal_k)
labels = clustering.fit_predict(X)


# PCAで2次元に圧縮
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


# クラスタの名前を定義
cluster_names = [f'Cluster {i+1}' for i in range(len(set(labels)) - (1 if -1 in labels else 0))]


# 2次元プロット
plt.figure()
for i in range(len(cluster_names)):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=cluster_names[i])
plt.title(f'{method_name} Clustering (2D projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# クラスタ結果を保存
sums = []
for i in range(len(cluster_names)):
    temp = [loaded_object_names[l] for l in range(len(labels)) if labels[l] == i]
    sums.append(temp)
print(sums)


# sumsをJSONファイルに保存
def save_sums_to_json(sums, filename='sums_data.json'):
    with open(filename, 'w') as json_file:
        json.dump(sums, json_file, indent=4)


# JSONファイルの保存
output_filepath = "/content/drive/MyDrive/lastsum/sums_data.json"  # 出力するJSONファイルのパス
save_sums_to_json(sums, output_filepath)



先ほどの関数でベクトル化したものをこの関数で読み込み、複数あるクラスタリング手法から、好きなものを選んでクラスタリングすることができる。その際、シルエットスコアが特定の値以下になったところをクラスタ数とする。なお、極端にその数hが少なくならないように最低数が設けられている　今回なら8。なお、100クラスタまで試してから最適なものを選択している。流れとしては、一度いろんなクラスタ数でたくさんクラスタリングを試行したのち、その中でもっともよかったもので再度クラスタリングしてその結果をJSONに保存している。





