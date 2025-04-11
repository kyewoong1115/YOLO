import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(2)

fall_counter = {}  # 사람별 낙상 프레임 카운트

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints

        for idx, (box, kp) in enumerate(zip(boxes, keypoints)):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_w, bbox_h = x2 - x1, y2 - y1
            keypoint_xy = kp.xy.cpu().numpy()

            person_id = idx  # 여러명 처리(간단하게 idx로 구분)

            # 낙상 조건 점수 초기화
            score = 0

            # 관절 수 체크(10개 이상 잡혀야 점수 계산 시작)
            if len(keypoint_xy) >= 5:
                head_x, head_y = keypoint_xy[0]
                hip_x, hip_y = keypoint_xy[8]

                # 조건1. 머리보다 엉덩이가 위에 있음(점수 3점 부여)
                if head_y > hip_y:
                    score += 3

                # 조건2. 상체 기울기 (어깨 좌우 거리)(상체 기울어짐 심하면 2점 부여)
                if len(keypoint_xy) > 6:
                    l_shoulder = keypoint_xy[5]
                    r_shoulder = keypoint_xy[6]
                    shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder)
                    height = abs(head_y - hip_y)
                    if height != 0:
                        angle_ratio = shoulder_dist / height
                        if angle_ratio > 1.2:  # 상체 기울기 심함
                            score += 2

                # 조건3. 바운딩 박스 비율(1보다 작으면 누워있다는 뜻으로 2점 부여)
                aspect_ratio = bbox_h / bbox_w
                if aspect_ratio < 1.0:
                    score += 2

                # 조건4. 바운딩 박스 크기(사람이 너무 멀리 있는 경우 200보다 작아짐, 작은 물체 오탐 가능성 존재,200이상 일때 1점 부여)
                if bbox_h > 200:
                    score += 1

                # 낙상 판별(6점 이상일때 낙상이라고 판단)
                if score >= 6:
                    fall_counter[person_id] = fall_counter.get(person_id, 0) + 1
                else:
                    fall_counter[person_id] = 0

                # 일정 프레임 이상 유지 시 낙상 처리
                if fall_counter[person_id] >= 10:  # 10프레임 유지
                    cv2.putText(annotated_frame, "FALL DETECTED!", (x1, y1 - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # 낙상 점수 표시
            cv2.putText(annotated_frame, f"Score: {score}", (x1, y2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


             # 바운딩 박스 표시
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('YOLO Pose Fall Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
