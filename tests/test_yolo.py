from ultralytics import YOLO


def test_yolo_inference_smoke():
    model = YOLO("best.pt")
    results = model.predict(
        source="example_test/colon_polyp.jpg",
        conf=0.25,
        device="cpu",
        verbose=False,
    )
    assert len(results) == 1
