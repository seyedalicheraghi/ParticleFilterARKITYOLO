import cv2


# Flags to be shared with callback functions
while_wait_for_click = True
click_location = None


# Callback function used to capture the image coordinates
# from a click on a displayed image
def mouse_get_location(event, x, y, flags, param):
    global click_location
    global while_wait_for_click
    if event == cv2.EVENT_LBUTTONDOWN:
        click_location = (x, y)
        while_wait_for_click = False
    if event == cv2.EVENT_RBUTTONDOWN:
        while_wait_for_click = False


def get_location_in_image(img):
    # show image
    cv2.imshow('Click Start Location', img)
    cv2.setMouseCallback('Click Start Location', mouse_get_location)
    while while_wait_for_click:
        cv2.waitKey(50)
    cv2.destroyWindow('Click Start Location')
    cv2.waitKey(1)
    return click_location
