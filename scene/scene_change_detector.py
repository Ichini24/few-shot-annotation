from typing import Optional
import numpy as np
import cv2


class SceneModelDifference:
    def __init__(self, min_frames_diff = 10):
        self._min_frames_diff = min_frames_diff
        self._allowance_counter = self._min_frames_diff

    def update_allowance_counter(self):
        self._allowance_counter = self._min_frames_diff

    def process(self, frame) -> bool:
        decision = self._process_internal(frame)

        if decision:
            self.update_allowance_counter()

        return decision

    def _process_internal(self, frame) -> bool:
        pass


class SceneChangeDetector:
    def __init__(self,
                 min_frames_diff,
                 flow_activation_threshold = 0.1,
                 flow_resize_size = None,
                 model: Optional[SceneModelDifference] = None):
        """
        Initializes scene change detector

        Args:
            min_frames_diff: min count between frames for activation.
             Even if there are several frames with activation in a row,
             only 'i' and 'i' + min_frames_diff will be activated.
            flow_activation_threshold: the relative activation area threshold.
            flow_resize_size: the resize to apply before calculating optical flow.
            Decrease if you want to perform the scene detection faster.
            model: the SceneModelDifference instance. May be anything(algorithm, ml algorithm, dnn, etc.),
            the interface is defined in SceneModelDifference base class
        """
        self._min_frames_diff = min_frames_diff
        self._allowance_counter = self._min_frames_diff
        self._flow_resize_size = flow_resize_size

        self._flow_activation_threshold = flow_activation_threshold

        self._previous_diff_gray = None
        self._previous_gray = None
        self._hsv = None

        self._model: Optional[SceneModelDifference] = model

    def process(self, frame) -> bool:
        model_result = False

        self._allowance_counter = max(self._allowance_counter - 1, 0)

        if self._allowance_counter > 0:
            return False

        if self._model:
            model_result = self._model.process(frame)

        optical_flow_result = self._process_optical_flow(frame)

        decision = (model_result or optical_flow_result) and self._allowance_counter == 0

        if decision:
            self._allowance_counter = self._min_frames_diff

            if self._model:
                self._model.update_allowance_counter()

        return decision

    def _preprocess_img(self, frame):
        masked = frame

        if self._previous_diff_gray is not None:
            diff_gray = cv2.absdiff(self._previous_diff_gray, frame)
            mask = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)[1]
            masked = cv2.bitwise_and(frame, frame, mask=mask)

        self._previous_diff_gray = frame

        return masked

    def _process_optical_flow(self, input_frame) -> bool:
        if self._flow_resize_size:
            frame = cv2.resize(input_frame, self._flow_resize_size)
        else:
            frame = input_frame.copy()

        frame = cv2.blur(frame, (5, 5))

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = self._preprocess_img(current_gray)

        activation = False

        if self._hsv is None:
            self._hsv = np.zeros_like(frame)
            self._hsv[..., 1] = 255
        else:
            flow = cv2.calcOpticalFlowFarneback(self._previous_gray,
                                                current_gray,
                                                None,
                                                0.5,
                                                2,
                                                15,
                                                3,
                                                5,
                                                1.1,
                                                0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            self._hsv[..., 0] = ang * 180 / np.pi / 2
            self._hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            activation = self._analyze_flow(self._hsv)

        self._previous_gray = current_gray
        return activation

    def _analyze_flow(self, flow):
        chans = cv2.split(self._hsv)
        magnitude = chans[2]
        filtered_magnitude = cv2.threshold(magnitude, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]

        non_zero_count = cv2.countNonZero(filtered_magnitude)
        total_pixels = filtered_magnitude.shape[0] * filtered_magnitude.shape[1]

        return non_zero_count / total_pixels > self._flow_activation_threshold
