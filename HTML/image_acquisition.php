<?php
// 图像采集页面 - 用于线上拍摄人脸
date_default_timezone_set('Asia/Kuala_Lumpur');
session_start();
include 'config.php';
include 'includes/log-functions.php';
include 'includes/notification-functions.php';
include 'includes/language-loader.php';

// 检查用户是否已登录
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header("Location: login.php");
    exit;
}

// 处理表单提交
$saved_files = [];
$saved_count = 0;

// 如果是从表单提交后重定向过来的，使用通知系统显示成功消息
if (isset($_GET['success']) && isset($_SESSION['form_submitted'])) {
    // 使用 set_notification 函数设置通知，将由 notifications-loader.php 处理
    set_success_notification(isset($_SESSION['success_message']) ? $_SESSION['success_message'] : __('images_saved_successfully_generic'));
    // 清除会话中的标志和消息，防止刷新页面时重复显示成功消息
    unset($_SESSION['form_submitted']);
    unset($_SESSION['success_message']);
}

// 获取当前登录用户信息
$current_username = $_SESSION['username'];
$current_user_id = $_SESSION['id'];

// 尝试从数据库获取用户的全名和头像
try {
    // 使用config.php中已有的数据库连接
    // $conn 是在config.php中创建的mysqli连接

    // 查询当前用户的全名和头像
    $stmt = $conn->prepare("SELECT fullname, profile_picture FROM users WHERE username = ?");
    $stmt->bind_param("s", $current_username);
    $stmt->execute();
    $result = $stmt->get_result();
    $user_data = $result->fetch_assoc();
    $stmt->close();

    if ($user_data && !empty($user_data['fullname'])) {
        $current_user_fullname = $user_data['fullname'];
        $current_user_profile_picture = $user_data['profile_picture'];
    } else {
        // 如果在数据库中找不到，则使用会话中的全名或用户名
        $current_user_fullname = $_SESSION['fullname'] ?? $current_username;
        $current_user_profile_picture = null;
    }
} catch (Exception $e) {
    // 数据库查询失败，使用会话中的全名或用户名
    error_log("Database error: " . $e->getMessage());
    $current_user_fullname = $_SESSION['fullname'] ?? $current_username;
    $current_user_profile_picture = null;
}

$is_admin = ($_SESSION['role'] ?? '') === 'admin';

// 检查用户目录是否存在
$user_dir_exists = false;
$dataset_dir = '/FRAS/Image_DataSet';
$alt_dirs = ['../Image_DataSet', 'Image_DataSet', '/var/www/html/Image_DataSet'];
$user_dir = '';
$image_count = 0;
$next_image_number = 1;

// 记录当前工作目录和用户名，帮助调试
$current_dir = getcwd();
error_log("Current working directory: " . $current_dir);
error_log("Checking directories for user: " . $current_user_fullname);

// 检查主路径
if (file_exists($dataset_dir) && is_dir($dataset_dir)) {
    $user_dir = $dataset_dir . '/' . $current_user_fullname;
    error_log("Checking primary path: " . $user_dir);
    if (file_exists($user_dir) && is_dir($user_dir)) {
        $user_dir_exists = true;
        error_log("User directory found in primary path");
    }
}

// 检查备选路径
if (!$user_dir_exists) {
    foreach ($alt_dirs as $alt_dir) {
        $full_alt_dir = realpath($alt_dir);
        if ($full_alt_dir && is_dir($full_alt_dir)) {
            $user_dir = $full_alt_dir . '/' . $current_user_fullname;
            error_log("Checking alternative path: " . $user_dir);
            if (file_exists($user_dir) && is_dir($user_dir)) {
                $user_dir_exists = true;
                error_log("User directory found in alternative path: " . $full_alt_dir);
                break;
            }
        }
    }
}

// 如果仍未找到，尝试直接在当前目录下查找
if (!$user_dir_exists) {
    $local_dir = 'Image_DataSet/' . $current_user_fullname;
    error_log("Checking local path: " . $local_dir);
    if (file_exists($local_dir) && is_dir($local_dir)) {
        $user_dir = $local_dir;
        $user_dir_exists = true;
        error_log("User directory found in local path");
    }
}

// 如果目录存在，计算图片数量和下一个编号
if ($user_dir_exists && !empty($user_dir) && is_dir($user_dir)) {
    $files = scandir($user_dir);
    $image_files = [];
    $image_numbers = [];

    // 计算图片数量并提取编号
    foreach ($files as $file) {
        if (preg_match('/\.(jpg|jpeg|png)$/i', $file)) {
            $image_files[] = $file;

            // 提取编号
            if (preg_match('/^' . preg_quote($current_user_fullname, '/') . '_(\d+)_/', $file, $matches)) {
                if (isset($matches[1]) && is_numeric($matches[1])) {
                    $image_numbers[] = (int)$matches[1];
                }
            }
        }
    }

    $image_count = count($image_files);

    // 检查是否达到25张图片的限制
    $number_exceeded = ($image_count >= 25);

    // 确定下一个编号
    if (!empty($image_numbers)) {
        $next_image_number = max($image_numbers) + 1;

        // 如果编号超过25但总数未达到25，则重置为1
        if ($next_image_number > 25 && !$number_exceeded) {
            $next_image_number = 1;
        }
    }

    error_log("User directory exists for $current_user_fullname with $image_count images, next number: $next_image_number");
}

// 获取所有用户列表（仅管理员可用）
$all_users = [];
if ($is_admin) {
    // 从数据库获取用户列表
    try {
        // 使用config.php中已有的数据库连接

        // 查询所有用户
        $stmt = $conn->prepare("SELECT username, fullname FROM users ORDER BY fullname");
        $stmt->execute();
        $result = $stmt->get_result();

        // 获取所有用户
        $all_users = [];
        while ($row = $result->fetch_assoc()) {
            $all_users[] = $row;
        }
        $stmt->close();

        // 如果查询失败，只添加当前用户
        if (empty($all_users)) {
            $all_users = [
                [
                    'username' => $current_username,
                    'fullname' => $current_user_fullname
                ]
            ];
        }
    } catch (Exception $e) {
        // 数据库查询失败，只添加当前用户
        error_log("Database error: " . $e->getMessage());

        // 只添加当前用户
        $all_users = [
            [
                'username' => $current_username,
                'fullname' => $current_user_fullname
            ]
        ];
    }
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    if (isset($_POST['save_images']) && !empty($_POST['all_images'])) {
        try {
            // 获取用户输入的姓名
            $fullname = trim($_POST['fullname']);
            if (empty($fullname)) {
                throw new Exception(__('name_required'));
            }

            // 安全检查：非管理员用户只能为自己拍照
            if (!$is_admin && $fullname !== $current_user_fullname) {
                error_log("Security warning: Non-admin user {$current_username} tried to save photos for {$fullname}");
                throw new Exception(__('security_error_non_admin_user'));
            }

            // 获取所有图像数据
            $all_images_json = $_POST['all_images'];
            $all_images = json_decode($all_images_json, true);

            // 检查用户目录是否存在 - 使用与页面加载时相同的逻辑检查所有可能的路径
            $user_dir_exists = false;
            $user_dir = '';

            // 检查主路径
            if (file_exists($dataset_dir) && is_dir($dataset_dir)) {
                $temp_dir = $dataset_dir . '/' . $fullname;
                if (file_exists($temp_dir) && is_dir($temp_dir)) {
                    $user_dir_exists = true;
                    $user_dir = $temp_dir;
                    error_log("POST: User directory found in primary path: " . $temp_dir);
                }
            }

            // 检查备选路径
            if (!$user_dir_exists) {
                $alt_dirs = ['../Image_DataSet', 'Image_DataSet', '/var/www/html/Image_DataSet'];
                foreach ($alt_dirs as $alt_dir) {
                    $full_alt_dir = realpath($alt_dir);
                    if ($full_alt_dir && is_dir($full_alt_dir)) {
                        $temp_dir = $full_alt_dir . '/' . $fullname;
                        if (file_exists($temp_dir) && is_dir($temp_dir)) {
                            $user_dir_exists = true;
                            $user_dir = $temp_dir;
                            error_log("POST: User directory found in alternative path: " . $temp_dir);
                            break;
                        }
                    }
                }
            }

            // 如果仍未找到，尝试直接在当前目录下查找
            if (!$user_dir_exists) {
                $local_dir = 'Image_DataSet/' . $fullname;
                if (file_exists($local_dir) && is_dir($local_dir)) {
                    $user_dir_exists = true;
                    $user_dir = $local_dir;
                    error_log("POST: User directory found in local path: " . $local_dir);
                }
            }

            // 如果目录不存在，需要至少10张照片；如果目录已存在，至少需要1张照片
            $min_required = $user_dir_exists ? 1 : 10;

            error_log("POST: User directory exists: " . ($user_dir_exists ? 'true' : 'false') . ", path: " . $user_dir);

            if (!is_array($all_images)) {
                throw new Exception("无效的图像数据");
            }

            if (count($all_images) < 1) {
                throw new Exception("请至少拍摄1张照片");
            }

            // 检查用户目录中已有的图片数量
            $existing_image_count = 0;
            if ($user_dir_exists && !empty($user_dir) && is_dir($user_dir)) {
                $files = scandir($user_dir);
                foreach ($files as $file) {
                    if (preg_match('/\.(jpg|jpeg|png)$/i', $file)) {
                        $existing_image_count++;
                    }
                }
                error_log("POST: Found {$existing_image_count} existing images in {$user_dir}");
            }

            // 验证逻辑
            error_log("POST: Validating - user_dir_exists: " . ($user_dir_exists ? 'true' : 'false') .
                      ", existing_image_count: " . $existing_image_count .
                      ", new images: " . count($all_images));

            // 检查是否会超过最大限制（25张）
            if ($user_dir_exists && ($existing_image_count + count($all_images)) > 25) {
                $allowed_new = 25 - $existing_image_count;
                if ($allowed_new <= 0) {
                    throw new Exception("用户图片数量已达到上限（25张），无法继续添加");
                } else {
                    throw new Exception("用户已有{$existing_image_count}张照片，最多只能再添加{$allowed_new}张，请减少上传数量");
                }
            }

            // 如果用户目录不存在，需要至少10张照片
            if (!$user_dir_exists && count($all_images) < 10) {
                throw new Exception("新用户需要至少10张照片");
            }
            // 如果用户目录存在且已有10张或以上照片，只需要1张新照片
            else if ($user_dir_exists && $existing_image_count >= 10 && count($all_images) < 1) {
                throw new Exception("已有用户至少需要1张新照片");
            }
            // 如果用户目录存在但图片少于10张，需要凑够10张
            else if ($user_dir_exists && $existing_image_count < 10 && (count($all_images) + $existing_image_count) < 10) {
                $needed = 10 - $existing_image_count;
                throw new Exception("当前用户只有{$existing_image_count}张照片，还需要至少{$needed}张照片才能达到10张的要求");
            }

            // 设置保存路径
            $dataset_dir = '/FRAS/Image_DataSet';

            // 检查主路径是否可写
            if (!is_dir($dataset_dir) || !is_writable($dataset_dir)) {
                error_log("Primary path not writable, trying alternative paths");

                // 尝试相对路径
                $alt_dataset_dir = '../Image_DataSet';
                if (is_dir($alt_dataset_dir) && is_writable($alt_dataset_dir)) {
                    $dataset_dir = $alt_dataset_dir;
                    error_log("Using alternative path: " . $alt_dataset_dir);
                } else {
                    // 尝试在当前目录创建
                    $alt_dataset_dir = 'Image_DataSet';
                    if (!is_dir($alt_dataset_dir)) {
                        mkdir($alt_dataset_dir, 0777, true);
                    }
                    if (is_dir($alt_dataset_dir) && is_writable($alt_dataset_dir)) {
                        $dataset_dir = $alt_dataset_dir;
                        error_log("Using local path: " . $alt_dataset_dir);
                    } else {
                        // 最后尝试系统临时目录
                        $dataset_dir = sys_get_temp_dir() . '/Image_DataSet';
                        error_log("Using temp directory: " . $dataset_dir);
                    }
                }
            }

            // 如果已经找到了用户目录，优先使用它
            if ($user_dir_exists && !empty($user_dir)) {
                $person_dir = $user_dir;
                error_log("POST: Using existing user directory: " . $person_dir);
            } else {
                $person_dir = $dataset_dir . '/' . $fullname;
                error_log("POST: Creating new user directory: " . $person_dir);
            }

            // 确保目录存在
            try {
                if (!file_exists($dataset_dir)) {
                    error_log("Creating dataset directory: " . $dataset_dir);
                    $result = mkdir($dataset_dir, 0777, true);
                    if (!$result) {
                        error_log("Failed to create dataset directory: " . $dataset_dir);
                        $error_details = error_get_last();
                        $error_msg = $error_details ? $error_details['message'] : 'Unknown error';
                        error_log("Error: " . $error_msg);

                        // 记录错误日志到系统日志
                        $log_message = "Failed to create dataset directory for face images: $dataset_dir. Error: $error_msg";
                        log_error($log_message, "image_acquisition.php");
                    } else {
                        // 确保权限设置正确
                        chmod($dataset_dir, 0777);
                        error_log("Successfully created dataset directory with permissions 0777");
                    }
                }

                if (!file_exists($person_dir)) {
                    error_log("Creating person directory: " . $person_dir);
                    $result = mkdir($person_dir, 0777, true);
                    if (!$result) {
                        error_log("Failed to create person directory: " . $person_dir);
                        $error_details = error_get_last();
                        $error_msg = $error_details ? $error_details['message'] : 'Unknown error';
                        error_log("Error: " . $error_msg);

                        // 记录错误日志到系统日志
                        $log_message = "Failed to create person directory for face images: $person_dir. User: $fullname. Error: $error_msg";
                        log_error($log_message, "image_acquisition.php");
                    } else {
                        // 确保权限设置正确
                        chmod($person_dir, 0777);
                        error_log("Successfully created person directory with permissions 0777");
                    }
                }
            } catch (Exception $e) {
                error_log("Exception while creating directories: " . $e->getMessage());
                throw new Exception("无法创建必要的目录: " . $e->getMessage());
            }

            // 保存所有图像
            foreach ($all_images as $index => $image_data) {
                // 解码base64图像数据
                $image_parts = explode(";base64,", $image_data);
                if (count($image_parts) !== 2) {
                    error_log("Invalid image data format (base64 part missing) for image " . $index);
                    continue;
                }

                $image_data_base64 = $image_parts[1];
                $decoded_data = base64_decode($image_data_base64);

                if ($decoded_data === false) {
                    error_log("Failed to decode base64 data for image " . $index);
                    continue;
                }

                // 获取目录中已有的图片数量，并确定下一个编号
                $next_number = 1;
                $image_count = 0;

                // 检查目录中的文件
                if (is_dir($person_dir)) {
                    $files = scandir($person_dir);
                    $existing_images = [];
                    $all_images = [];

                    // 提取所有图片文件名中的编号
                    foreach ($files as $file) {
                        if (preg_match('/\.(jpg|jpeg|png)$/i', $file)) {
                            $all_images[] = $file;

                            if (preg_match('/^' . preg_quote($fullname, '/') . '_(\d+)_/', $file, $matches)) {
                                if (isset($matches[1]) && is_numeric($matches[1])) {
                                    $existing_images[] = (int)$matches[1];
                                }
                            }
                        }
                    }

                    // 这里的$image_count是已有的图片数量
                    // 新上传的图片数量
                    $new_images_count = count($all_images);
                    // 上传后的总图片数量
                    $total_after_upload = $image_count + $new_images_count;

                    error_log("Checking limits - Current images: $image_count, New images: $new_images_count, Total after upload: $total_after_upload");

                    if ($image_count >= 25) {
                        error_log("Error: User has reached the maximum limit of 25 images");
                        throw new Exception("用户图片数量已达到上限（25张），无法继续添加");
                    } else if ($total_after_upload > 25) {
                        $allowed_new = 25 - $image_count;
                        error_log("Error: Adding these images would exceed the limit of 25. Current: $image_count, New: $new_images_count");
                        throw new Exception("用户已有{$image_count}张照片，最多只能再添加{$allowed_new}张，请减少上传数量");
                    }

                    // 如果找到了编号，确定下一个编号
                    if (!empty($existing_images)) {
                        $next_number = max($existing_images) + 1;
                    }

                    error_log("Found " . $image_count . " existing images, next number: " . $next_number);
                }

                // 如果编号超过25但总数未达到25，则重置为1
                if ($next_number > 25 && $image_count < 25) {
                    error_log("Warning: Image number exceeds 25, resetting to 1");
                    $next_number = 1;
                }

                // 创建文件名 - 使用用户的全名 + 号码(两位数) + 当前日期和时间
                $timestamp = date('Ymd_His'); // 格式化为年月日_时分秒
                $filename = $fullname . '_' . sprintf('%02d', $next_number) . '_' . $timestamp . '.jpg';
                // 只替换文件名中的特殊字符，保留空格
                $filename = preg_replace('/[^a-zA-Z0-9_\-\. ]/', '', $filename);
                $filepath = $person_dir . '/' . $filename;

                // 记录文件名信息
                error_log("Creating image with filename: " . $filename);

                // 保存图片到文件系统
                $result = file_put_contents($filepath, $decoded_data);

                if ($result !== false) {
                    // 确保文件权限正确
                    chmod($filepath, 0777); // 设置为所有用户可读写执行

                    // 强制文件系统同步，确保更改立即可见
                    clearstatcache(true, $filepath);

                    // 验证保存的文件
                    if (file_exists($filepath) && filesize($filepath) > 0) {
                        // 记录日志
                        if ($saved_count == 0) {
                            $log_message = "Face images captured for: $fullname";
                            log_info($log_message, "image_acquisition.php");
                        }
                        error_log("Successfully saved image to: " . $filepath . " (Size: " . $result . " bytes)");

                        // 记录已保存的文件
                        $saved_files[] = $filepath;
                        $saved_count++;
                    } else {
                        error_log("File exists but size is 0 or file doesn't exist after saving");
                    }
                } else {
                    error_log("Failed to save image to: " . $filepath);
                    // 记录错误日志到系统日志
                    $error_message = "Failed to save face image for user: $fullname, path: $filepath";
                    log_error($error_message, "image_acquisition.php");
                }
            }

            if ($saved_count > 0) {
                // 将成功消息保存到会话中，以便在重定向后显示
                $_SESSION['success_message'] = sprintf(__('images_saved_successfully'), $saved_count, $fullname);

                // 设置一个标志，表示表单已成功提交
                $_SESSION['form_submitted'] = true;

                // 重定向到同一页面，避免表单重复提交
                header("Location: " . $_SERVER['PHP_SELF'] . "?success=1");
                exit;
            } else {
                throw new Exception(__('no_images_saved'));
            }

        } catch (Exception $e) {
            // 使用通知系统显示错误消息
            set_error_notification($e->getMessage());
            error_log("Error in image_acquisition.php: " . $e->getMessage());
        }
    }
}
?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <link rel="stylesheet" href="css/dashboard-style.css">
    <link rel="stylesheet" href="css/slidebar.css">
    <link rel="stylesheet" href="css/image-acquisition-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        /* 人脸状态显示样式 */
        #face-status {
            position: absolute;
            bottom: 10px;
            left: 10px;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: bold;
            z-index: 100;
            transition: all 0.3s ease;
            opacity: 0.9;
        }
        .face-status-hidden {
            display: none !important; /* 使用 !important 确保隐藏优先级最高 */
            visibility: hidden !important;
            opacity: 0 !important;
        }
        .face-status-ok {
            display: block;
            background-color: rgba(117, 183, 67, 0.8);
            color: white;
        }
        .face-status-warning {
            display: block;
            background-color: rgba(255, 152, 0, 0.8);
            color: white;
        }
        .face-status-error {
            display: block;
            background-color: rgba(197, 26, 74, 0.8);
            color: white;
        }
        /* 按钮样式 */
        .button-danger {
            background-color: #C51A4A;
            color: white;
        }
        .button-danger:hover {
            background-color: #a01540;
        }
        /* 置信度显示样式 - 移除，将在JavaScript中动态创建 */
    </style>
    <?php
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    include 'includes/notifications-loader.php';
    include 'includes/slidebar.php';
    ?>
    <title><?php echo __('face_image_acquisition'); ?> - FRAS System</title>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/face_detection.js" crossorigin="anonymous"></script>

    <!-- 添加辅助函数，用于替代alert，使用通知系统 -->
    <script>
        // 显示通知而不是alert
        function showNotificationAlert(message, type = 'error') {
            // 如果通知函数存在，使用通知系统
            if (typeof showNotification === 'function') {
                if (type === 'error') {
                    showErrorNotification(message);
                } else if (type === 'warning') {
                    showWarningNotification(message);
                } else if (type === 'success') {
                    showSuccessNotification(message);
                } else {
                    showNotification(message, type);
                }
            } else {
                // 如果通知系统不可用，回退到alert
                alert(message);
            }
        }
    </script>
</head>
<body>
    <?php include 'includes/sidebar.php'; ?>

    <div class="main-content">
        <div class="container">
            <!-- Header -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('face_image_acquisition'); ?></h2>

                <div class="user-profile">
                    <img src="data:image/jpeg;base64,<?php echo base64_encode($current_user_profile_picture); ?>" alt="Profile Picture">
                </div>
            </div>

            <div class="camera-container">
                <!-- 左侧摄像头区域 -->
                <div class="camera-left-column">
                    <div class="video-container">
                        <video id="video" autoplay playsinline></video>
                        <div id="face-status" class="face-status-hidden">
                            <span id="face-status-text"><?php echo __('waiting_for_face_detection'); ?></span>
                        </div>
                    </div>

                    <!-- 进度条和按钮的统一容器 -->
                    <div class="controls-container">
                        <div class="capture-progress-container">
                            <div class="progress-bar">
                                <div class="progress" id="capture-progress" style="width: 0%"></div>
                            </div>
                            <div class="progress-text">
                                <span id="captured-count">0</span> / <span id="required-count">10</span> <?php echo __('photos'); ?>
                                <span id="photo-requirement-text"><?php echo str_replace('{count}', '10', __('new_user_requires_photos')); ?></span>
                            </div>
                        </div>

                        <div class="button-group">
                            <button type="button" id="start-camera" class="button button-secondary">
                                <i class="fas fa-video"></i> <?php echo __('start_camera'); ?>
                            </button>
                            <button type="button" id="capture-button" class="button button-primary" disabled>
                                <i class="fas fa-camera"></i> <?php echo __('capture'); ?>
                            </button>
                            <button type="button" id="upload-button" class="button button-primary">
                                <i class="fas fa-upload"></i> <?php echo __('upload'); ?>
                            </button>
                            <input type="file" id="image-upload-input" accept="image/*" multiple style="display: none;">
                            <button type="submit" form="capture-form" id="save-button" class="button button-primary" disabled>
                                <i class="fas fa-save"></i> <?php echo __('save'); ?>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- 右侧信息区域 -->
                <div class="camera-right-column">
                    <div class="form-section">
                        <form id="capture-form" method="post" onsubmit="return validateForm()">
                            <div class="form-group">
                                <label for="fullname"><?php echo __('full_name'); ?>: <span class="required">*</span></label>
                                <?php if ($is_admin): ?>
                                    <select id="fullname" name="fullname" required>
                                        <option value="<?php echo htmlspecialchars($current_user_fullname); ?>"><?php echo htmlspecialchars($current_user_fullname); ?></option>
                                        <?php foreach ($all_users as $user): ?>
                                            <?php if ($user['fullname'] != $current_user_fullname): ?>
                                                <option value="<?php echo htmlspecialchars($user['fullname']); ?>"><?php echo htmlspecialchars($user['fullname']); ?></option>
                                            <?php endif; ?>
                                        <?php endforeach; ?>
                                    </select>
                                    <script>
                                        // 在页面加载时检查是否有保存的用户选择
                                        document.addEventListener('DOMContentLoaded', function() {
                                            const savedUser = localStorage.getItem('selectedUser');
                                            if (savedUser) {
                                                const fullnameSelect = document.getElementById('fullname');
                                                // 检查保存的用户是否在选项列表中
                                                for (let i = 0; i < fullnameSelect.options.length; i++) {
                                                    if (fullnameSelect.options[i].value === savedUser) {
                                                        fullnameSelect.selectedIndex = i;
                                                        break;
                                                    }
                                                }
                                            }
                                        });
                                    </script>
                                <?php else: ?>
                                    <input type="text" id="fullname" name="fullname" value="<?php echo htmlspecialchars($current_user_fullname); ?>" readonly>
                                <?php endif; ?>
                            </div>

                            <div id="user-dir-status" class="<?php echo $user_dir_exists ? ($image_count >= 25 ? 'status-error' : 'status-success') : 'status-warning'; ?>">
                                <?php if ($user_dir_exists): ?>
                                    <?php if ($image_count >= 25): ?>
                                        <i class="fas fa-ban"></i>
                                        <?php echo sprintf(__('user_image_limit_reached_25'), $image_count); ?>
                                    <?php else: ?>
                                        <i class="fas fa-check-circle"></i>
                                        <span class="user-dir-exists-text"><?php echo __('user_dir_exists_prefix') . $image_count . __('user_dir_exists_middle') . $next_image_number . __('user_dir_exists_suffix'); ?></span>
                                    <?php endif; ?>
                                <?php else: ?>
                                    <i class="fas fa-exclamation-triangle"></i> <?php echo __('user_dir_not_exists'); ?>
                                <?php endif; ?>
                            </div>

                            <!-- 移除重复的用户图片限制消息 -->

                            <input type="hidden" name="image_data" id="image_data">
                            <input type="hidden" name="all_images" id="all_images">
                            <input type="hidden" name="save_images" value="1">
                            <input type="hidden" id="user_dir_exists" value="<?php echo $user_dir_exists ? '1' : '0'; ?>">
                        </form>
                    </div>

                    <!-- 图片计数器 -->
                    <div class="image-counter">
                        <div class="image-counter-title">
                            <i class="fas fa-images"></i> <?php echo __('image_count'); ?>
                        </div>
                        <div class="counter-details">
                            <div class="counter-item">
                                <span class="counter-label"><?php echo __('existing_photos'); ?>:</span>
                                <span class="counter-value" id="existing-count-display">0</span>
                            </div>
                            <div class="counter-item">
                                <span class="counter-label"><?php echo __('new_photos'); ?>:</span>
                                <span class="counter-value" id="new-count-display">0</span>
                            </div>
                            <div class="counter-item">
                                <span class="counter-label"><?php echo __('total_after_upload'); ?>:</span>
                                <span class="counter-value" id="total-count-display">0 / 25</span>
                            </div>
                        </div>
                    </div>

                    <!-- 已捕获的图像预览 -->
                    <div class="preview-section">
                        <h3><i class="fas fa-camera"></i> <?php echo __('captured_images'); ?></h3>
                        <div class="preview-container" id="preview-container"></div>
                    </div>

                    <!-- 隐藏的总计数容器，用于兼容现有JavaScript -->
                    <div class="total-count-text" id="total-count-container" style="display: none;">
                        <?php echo __('existing_photos'); ?>: <span id="existing-count">0</span> +
                        <?php echo __('new_photos'); ?>: <span id="new-count">0</span> =
                        <?php echo __('total_after_upload'); ?>: <span id="total-count">0</span> / 25
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 全局变量
        let stream = null;
        let video = document.getElementById('video');
        let captureButton = document.getElementById('capture-button');
        let saveButton = document.getElementById('save-button');
        let startButton = document.getElementById('start-camera');
        let previewContainer = document.getElementById('preview-container');
        let capturedCount = document.getElementById('captured-count');
        let capturedImages = [];
        let faceDetected = false;
        let faceOccluded = false;
        let canvasElement = null;
        let canvasCtx = null;
        let faceDetector = null;
        let camera = null;
        let faceStatusElement = null;
        let lastDetectionResults = null; // 存储最新的人脸检测结果
        let cameraActive = false; // 跟踪摄像头状态
        let limitNotificationShown = false; // 跟踪是否已显示过图片数量达到上限的通知

        // 检查是否达到图片数量限制
        const imageCountLimitReached = <?php echo ($user_dir_exists && $image_count >= 25) ? 'true' : 'false'; ?>;

        // 如果达到限制，禁用拍照和保存按钮
        if (imageCountLimitReached) {
            captureButton.disabled = true;
            saveButton.disabled = true;
            startButton.disabled = true;
            startButton.title = "<?php echo __('user_image_limit_reached'); ?>";

            // 不设置通知标记，让DOMContentLoaded事件处理通知
        }

        // 页面加载完成后立即检查用户目录状态
        document.addEventListener('DOMContentLoaded', function() {
            // 不在这里显示通知，而是在updateUserDirectoryStatus函数中处理
            // 这样可以确保通知是基于最新的用户目录状态
            // 检查URL参数，如果是从表单提交后重定向过来的，则清空预览区域和表单数据
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.has('success')) {
                // 清空预览区域
                previewContainer.innerHTML = '';

                // 重置捕获的图像数组
                capturedImages = [];

                // 更新UI
                updateProgressUI();

                // 清空隐藏字段
                document.getElementById('image_data').value = '';
                document.getElementById('all_images').value = '';

                console.log('Form submitted successfully, cleared preview area and form data');

                // 注意：我们不清除localStorage中保存的用户选择，以便在表单提交后仍然保持选择的用户
            }

            // 设置初始图片数量数据属性
            const userDirStatus = document.getElementById('user-dir-status');
            userDirStatus.dataset.imageCount = <?php echo $image_count; ?>;

            // 初始化计数器显示
            const existingCountDisplay = document.getElementById('existing-count-display');
            const newCountDisplay = document.getElementById('new-count-display');
            const totalCountDisplay = document.getElementById('total-count-display');

            if (existingCountDisplay) existingCountDisplay.textContent = <?php echo $image_count; ?>;
            if (newCountDisplay) newCountDisplay.textContent = '0';
            if (totalCountDisplay) totalCountDisplay.textContent = `<?php echo $image_count; ?> / 25`;

            // 为当前用户检查目录状态，无论是管理员还是普通用户
            const fullnameInput = document.getElementById('fullname');
            if (fullnameInput && fullnameInput.value) {
                checkUserDirectoryStatus(fullnameInput.value);
            } else {
                // 如果没有用户名输入，直接更新UI
                updateProgressUI();
            }

            console.log('Page loaded, user directory exists: <?php echo $user_dir_exists ? 'true' : 'false'; ?>');
            console.log('Image count: <?php echo $image_count; ?>');
            console.log('Next image number: <?php echo $next_image_number; ?>');

            // 初始化总数计算显示
            updateProgressUI();
        });

        // 检查用户目录状态的函数
        function checkUserDirectoryStatus(username) {
            if (!username) return;

            const userDirStatus = document.getElementById('user-dir-status');

            // 显示加载中状态
            userDirStatus.className = 'status-warning';
            userDirStatus.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <?php echo __('checking_user_dir_status'); ?>';

            // 发送AJAX请求检查用户目录是否存在
            const xhr = new XMLHttpRequest();
            xhr.open('POST', 'image_acquisition-ajax.php', true);
            xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
            xhr.onreadystatechange = function() {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        console.log('AJAX response:', response);

                        // 更新UI
                        updateUserDirectoryStatus(response);
                    } catch (e) {
                        console.error('Error parsing JSON response:', e);

                        // 默认假设目录不存在
                        userDirStatus.className = 'status-warning';
                        userDirStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <?php echo __('user_dir_not_exists'); ?>';
                        document.getElementById('user_dir_exists').value = '0';
                        updateProgressUI();
                    }
                }
            };
            xhr.send('fullname=' + encodeURIComponent(username));
        }

        // 更新用户目录状态UI的函数
        function updateUserDirectoryStatus(response) {
            const userDirStatus = document.getElementById('user-dir-status');

            // 检查是否达到25张图片的限制
            if (response.exists) {
                if (response.number_exceeded) {
                    // 图片数量达到25张，锁定功能
                    userDirStatus.className = 'status-error';
                    userDirStatus.innerHTML = `
                        <i class="fas fa-ban"></i>
                        <?php echo __('user_image_limit_reached_25'); ?>
                    `;
                    document.getElementById('user_dir_exists').value = '1';
                    // 禁用拍照和保存按钮
                    captureButton.disabled = true;
                    saveButton.disabled = true;
                    startButton.disabled = true;
                    startButton.title = "<?php echo __('user_image_limit_reached_25'); ?>";

                    // 如果图片数量达到25张，且之前没有显示过通知，显示红色通知
                    if (!limitNotificationShown) {
                        setTimeout(() => {
                            showNotificationAlert("<?php echo __('user_image_limit_reached_25'); ?>", 'error');
                            limitNotificationShown = true;
                        }, 500);
                    }
                } else {
                    // 显示目录存在和图片数量信息
                    userDirStatus.className = 'status-success';
                    userDirStatus.innerHTML = `
                        <i class="fas fa-check-circle"></i>
                        <span class="user-dir-exists-text"><?php echo __('user_dir_exists_prefix'); ?> ${response.image_count} <?php echo __('user_dir_exists_middle'); ?> ${response.next_image_number}<?php echo __('user_dir_exists_suffix'); ?></span>
                    `;
                    document.getElementById('user_dir_exists').value = '1';
                    // 启用拍照按钮
                    if (video.srcObject) {
                        captureButton.disabled = false;
                    }
                    if (!cameraActive) {
                        startButton.disabled = false;
                        startButton.title = "";
                    }

                    // 如果图片数量未达到25张，重置通知标记
                    limitNotificationShown = false;
                }
            } else {
                userDirStatus.className = 'status-warning';
                userDirStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <?php echo __('user_dir_not_exists'); ?>';
                document.getElementById('user_dir_exists').value = '0';
                // 启用拍照按钮
                if (video.srcObject) {
                    captureButton.disabled = false;
                }
                if (!cameraActive) {
                    startButton.disabled = false;
                    startButton.title = "";
                }

                // 如果用户目录不存在，重置通知标记
                limitNotificationShown = false;
            }

            // 将图片数量保存到一个数据属性中，以便其他函数使用
            userDirStatus.dataset.imageCount = response.exists ? response.image_count : 0;

            // 更新计数器显示
            const existingCountDisplay = document.getElementById('existing-count-display');
            const totalCountDisplay = document.getElementById('total-count-display');

            if (existingCountDisplay) {
                existingCountDisplay.textContent = response.exists ? response.image_count : 0;
            }

            if (totalCountDisplay) {
                const existingCount = response.exists ? response.image_count : 0;
                totalCountDisplay.textContent = `${existingCount} / 25`;
            }

            // 更新进度条
            updateProgressUI();
        }

        // 需要拍摄的照片数量
        const REQUIRED_PHOTOS = 10;

        // 初始化人脸检测器
        async function initFaceDetector() {
            try {
                // 完全重新创建人脸检测器
                faceDetector = new FaceDetection({
                    locateFile: (file) => {
                        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
                    }
                });

                faceDetector.setOptions({
                    model: 'short', // 'short' 是轻量级模型，'full' 是更精确但更慢的模型
                    minDetectionConfidence: 0.85
                });

                // 等待加载完成
                await faceDetector.initialize();

                // 设置结果回调
                faceDetector.onResults(onResults);

                console.log('Face detector initialized successfully');
                return true;
            } catch (error) {
                console.error('Error initializing face detector:', error);
                return false;
            }
        }

        // 启动摄像头
        async function startCamera() {
            try {
                // 创建覆盖在视频上的画布用于绘制人脸框
                if (!canvasElement) {
                    canvasElement = document.createElement('canvas');
                    canvasElement.className = 'output_canvas';
                    canvasElement.style.position = 'absolute';
                    canvasElement.style.left = '0';
                    canvasElement.style.top = '0';
                    canvasElement.style.width = '100%';
                    canvasElement.style.height = '100%';
                    document.querySelector('.video-container').appendChild(canvasElement);
                    canvasCtx = canvasElement.getContext('2d');
                } else {
                    // 清除画布
                    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                }

                // 初始化人脸检测器
                const detectorInitialized = await initFaceDetector();
                if (!detectorInitialized) {
                    throw new Error('Failed to initialize face detector');
                }

                // 获取摄像头流
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1920 },
                        height: { ideal: 1080 },
                        facingMode: 'user'
                    }
                });

                video.srcObject = stream;

                // 等待视频元素加载
                await new Promise((resolve) => {
                    video.onloadedmetadata = () => {
                        resolve();
                    };
                    // 如果已经加载完成，直接解析
                    if (video.readyState >= 2) {
                        resolve();
                    }
                });

                // 确保视频已经开始播放
                await video.play();

                // 使用 MediaPipe Camera 类处理视频帧
                camera = new Camera(video, {
                    onFrame: async () => {
                        if (canvasElement && video.videoWidth > 0 && video.videoHeight > 0 && cameraActive) {
                            canvasElement.width = video.videoWidth;
                            canvasElement.height = video.videoHeight;
                            try {
                                await faceDetector.send({image: video});
                            } catch (error) {
                                console.error('Error in face detection:', error);

                                // 使用AJAX记录人脸检测错误到系统日志
                                const errorMsg = error ? (error.message || JSON.stringify(error)) : 'Unknown error';
                                const logData = new FormData();
                                logData.append('action', 'log_error');
                                logData.append('message', 'Face detection error: ' + errorMsg);
                                logData.append('source', 'image_acquisition.php');

                                fetch('log-ajax.php', {
                                    method: 'POST',
                                    body: logData
                                }).catch(e => console.error('Failed to log error:', e));
                            }
                        }
                    },
                    width: 1920,
                    height: 1080
                });

                camera.start();
                cameraActive = true;

                console.log('Camera started successfully');
                return true;
            } catch (err) {
                console.error('Error starting camera:', err);
                return false;
            }
        }

        // 停止摄像头
        function stopCamera() {
            try {
                // 停止摄像头
                if (camera) {
                    camera.stop();
                    camera = null;
                }

                if (stream) {
                    stream.getTracks().forEach(track => {
                        track.stop();
                    });
                    stream = null;
                }

                // 清除视频源
                if (video) {
                    video.srcObject = null;
                }

                // 清除画布
                if (canvasCtx && canvasElement) {
                    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                }

                // 重置人脸检测状态
                faceDetected = false;
                faceOccluded = false;
                faceAreaTooSmall = true;
                lastDetectionResults = null;

                // 隐藏人脸状态显示 - 确保获取最新的元素引用
                faceStatusElement = document.getElementById('face-status');
                if (faceStatusElement) {
                    // 完全隐藏元素
                    faceStatusElement.style.display = 'none';
                    faceStatusElement.classList.add('face-status-hidden');
                    faceStatusElement.classList.remove('face-status-ok', 'face-status-warning', 'face-status-error');

                    // 清空状态文本
                    const statusText = document.getElementById('face-status-text');
                    if (statusText) {
                        statusText.textContent = '';
                    }
                }

                cameraActive = false;
                console.log('Camera stopped successfully');
                return true;
            } catch (error) {
                console.error('Error stopping camera:', error);
                return false;
            }
        }

        // 启动/停止摄像头按钮事件
        startButton.addEventListener('click', async () => {
            // 禁用按钮，防止重复点击
            startButton.disabled = true;

            try {
                // 如果摄像头已经启动，则停止摄像头
                if (cameraActive) {
                    // 停止摄像头
                    const stopped = stopCamera();

                    if (stopped) {
                        // 更新UI
                        captureButton.disabled = true;

                        // 更改按钮文本和图标
                        startButton.innerHTML = '<i class="fas fa-video"></i> <?php echo __('start_camera'); ?>';
                        startButton.classList.remove('button-danger');
                        startButton.classList.add('button-secondary');
                    } else {
                        showNotificationAlert('<?php echo __('failed_to_stop_camera'); ?>');
                    }
                } else {
                    // 启动摄像头
                    const started = await startCamera();

                    if (started) {
                        // 更改按钮文本和图标
                        startButton.innerHTML = '<i class="fas fa-stop"></i> <?php echo __('stop_camera'); ?>';
                        startButton.classList.remove('button-secondary');
                        startButton.classList.add('button-danger');
                    } else {
                        showNotificationAlert('<?php echo __('camera_access_error'); ?>');

                        // 记录摄像头访问错误到系统日志
                        const log_message = "Camera access error for user: " + "<?php echo addslashes($current_user_fullname); ?>";
                        console.error(log_message);
                    }
                }
            } catch (err) {
                console.error('Error in camera toggle:', err);
                showNotificationAlert('<?php echo __('error_in_camera_toggle'); ?>');

                // 记录摄像头切换错误到系统日志
                fetch('log-ajax.php', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: 'action=log_error&message=' + encodeURIComponent("Camera toggle error for user: <?php echo addslashes($current_user_fullname); ?>. Error: " + (err.message || "Unknown error")) + '&source=image_acquisition.php'
                }).catch(e => console.error('Failed to log error:', e));

                // 重置摄像头状态
                stopCamera();

                // 重置按钮状态
                startButton.innerHTML = '<i class="fas fa-video"></i> <?php echo __('start_camera'); ?>';
                startButton.classList.remove('button-danger');
                startButton.classList.add('button-secondary');
            } finally {
                // 重新启用按钮
                startButton.disabled = false;
            }
        });

        // 处理 MediaPipe 检测结果
        function onResults(results) {
            // 如果摄像头已停止或画布不存在，不处理结果
            if (!cameraActive || !canvasElement || !canvasCtx) {
                // 确保人脸状态显示被隐藏
                faceStatusElement = document.getElementById('face-status');
                if (faceStatusElement) {
                    faceStatusElement.style.display = 'none';
                    faceStatusElement.classList.add('face-status-hidden');
                    faceStatusElement.classList.remove('face-status-ok', 'face-status-warning', 'face-status-error');
                }
                return;
            }

            try {
                // 保存最新的检测结果，用于拍照时裁剪
                lastDetectionResults = results;

                // 清除画布
                canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

                // 获取人脸状态显示元素
                if (!faceStatusElement) {
                    faceStatusElement = document.getElementById('face-status');
                }

                // 检查是否检测到人脸
                if (results.detections && results.detections.length > 0) {
                    faceDetected = true;

                    // 处理第一个检测到的人脸
                    const detection = results.detections[0];

                    // 检查是否有遮挡
                    const hasOcclusion = checkFaceOcclusion(detection);
                    faceOccluded = hasOcclusion.occluded;

                    // 获取置信度值，用于计算人脸面积百分比，以便在人脸框上方绘制
                    const confidence = hasOcclusion.confidence;

                    // 计算人脸面积占画面的百分比
                    const boundingBox = detection.boundingBox;
                    const width = boundingBox.width * canvasElement.width;
                    const height = boundingBox.height * canvasElement.height;
                    const faceArea = width * height;
                    const totalArea = canvasElement.width * canvasElement.height;
                    const areaPercentage = (faceArea / totalArea) * 100;

                    // 检查人脸面积是否足够大（至少20%）
                    faceAreaTooSmall = areaPercentage < 20; // 更新全局变量

                    // 更新人脸状态显示，如果面积太小则显示提示
                    let statusReason = hasOcclusion.reason;
                    if (!hasOcclusion.occluded && faceAreaTooSmall) {
                        statusReason = "<?php echo __('face_area_too_small_reason'); ?>";
                    }
                    updateFaceStatusDisplay(faceDetected, faceOccluded || faceAreaTooSmall, statusReason);

                    // 只有在没有遮挡且人脸面积足够大的情况下才启用拍照按钮
                    captureButton.disabled = faceOccluded || faceAreaTooSmall;

                    // 绘制人脸框
                    results.detections.forEach(detection => {
                        // 绘制边界框
                        const boundingBox = detection.boundingBox;
                        const width = boundingBox.width * canvasElement.width;
                        const height = boundingBox.height * canvasElement.height;
                        const x = boundingBox.xCenter * canvasElement.width - width / 2;
                        const y = boundingBox.yCenter * canvasElement.height - height / 2;

                        // 根据是否有遮挡或面积太小设置不同的颜色
                        canvasCtx.strokeStyle = (faceOccluded || faceAreaTooSmall) ? '#C51A4A' : '#75B743'; // 红色表示有遮挡或面积太小，绿色表示正常
                        canvasCtx.lineWidth = 7.5;
                        canvasCtx.strokeRect(x, y, width, height);

                        // 在人脸框上方绘制人脸面积百分比
                        const areaText = `${areaPercentage.toFixed(1)}%`;
                        canvasCtx.font = 'bold 36px Arial';
                        canvasCtx.textAlign = 'center';

                        // 绘制黑色描边
                        canvasCtx.lineWidth = 5; // 增加描边宽度以适应更大的字体
                        canvasCtx.strokeStyle = 'black';
                        canvasCtx.strokeText(areaText, x + width/2, y - 15); // 调整位置，使文本不会太靠近框

                        // 绘制白色文本
                        canvasCtx.fillStyle = 'white';
                        canvasCtx.fillText(areaText, x + width/2, y - 15);

                        // 绘制关键点
                        if (detection.keypoints) {
                            canvasCtx.fillStyle = (faceOccluded || faceAreaTooSmall) ? '#C51A4A' : '#75B743';
                            detection.keypoints.forEach(keypoint => {
                                canvasCtx.beginPath();
                                canvasCtx.arc(
                                    keypoint.x * canvasElement.width,
                                    keypoint.y * canvasElement.height,
                                    3, 0, 2 * Math.PI);
                                canvasCtx.fill();
                            });
                        }
                    });
                } else {
                    faceDetected = false;
                    faceOccluded = false;
                    faceAreaTooSmall = true; // 没有检测到人脸时，设置面积为太小

                    // 更新人脸状态显示
                    updateFaceStatusDisplay(faceDetected, faceOccluded);

                    // 禁用拍照按钮，因为没有检测到人脸
                    captureButton.disabled = true;
                }
            } catch (error) {
                console.error('Error in onResults function:', error);
                // 出错时不禁用拍照按钮，让用户可以尝试拍照
                captureButton.disabled = false;
            }
        }

        // 检查人脸是否有遮挡
        function checkFaceOcclusion(detection) {
            // 默认结果
            let result = {
                occluded: false,
                reason: "",
                confidence: 0
            };

            // 如果没有关键点，无法判断遮挡
            if (!detection.keypoints || detection.keypoints.length < 6) {
                // 尝试从detection.score获取置信度
                if (detection.score && detection.score.length > 0) {
                    result.confidence = detection.score[0];
                }
                return result;
            }

            // 获取关键点
            const keypoints = detection.keypoints;

            // 计算关键点的平均置信度
            let totalScore = 0;
            keypoints.forEach(keypoint => {
                totalScore += keypoint.score || 0;
            });
            const avgScore = totalScore / keypoints.length;

            // 保存置信度到结果中
            result.confidence = detection.score && detection.score.length > 0 ?
                detection.score[0] : avgScore;

            // 如果平均置信度过低，认为有遮挡
            if (avgScore < 0.85) {
                result.occluded = true;
                result.reason = "<?php echo __('face_keypoints_confidence_low'); ?>";
                return result;
            }

            // 检查眼睛区域
            const rightEye = keypoints.find(kp => kp.name === "rightEye");
            const leftEye = keypoints.find(kp => kp.name === "leftEye");

            if (rightEye && leftEye) {
                // 如果眼睛关键点的置信度低，可能被眼镜或其他物体遮挡
                if (rightEye.score < 0.85 || leftEye.score < 0.85) {
                    result.occluded = true;
                    result.reason = "<?php echo __('eyes_area_occluded'); ?>";
                    return result;
                }
            }

            // 检查鼻子和嘴巴区域
            const nose = keypoints.find(kp => kp.name === "nose");
            const mouth = keypoints.find(kp => kp.name === "mouth");

            if (nose && nose.score < 0.85) {
                result.occluded = true;
                result.reason = "<?php echo __('nose_area_occluded'); ?>";
                return result;
            }

            if (mouth && mouth.score < 0.85) {
                result.occluded = true;
                result.reason = "<?php echo __('mouth_area_occluded'); ?>";
                return result;
            }

            // 检查人脸框的宽高比，如果太窄或太宽，可能是侧脸
            const boundingBox = detection.boundingBox;
            const aspectRatio = boundingBox.width / boundingBox.height;

            if (aspectRatio < 0.5 || aspectRatio > 1.5) {
                result.occluded = true;
                result.reason = "<?php echo __('face_not_frontal'); ?>";
                return result;
            }

            // 检查人脸大小，如果太小，可能离摄像头太远
            if (boundingBox.width < 0.15 || boundingBox.height < 0.15) {
                result.occluded = true;
                result.reason = "<?php echo __('face_too_small'); ?>";
                return result;
            }

            return result;
        }

        // 更新人脸状态显示
        function updateFaceStatusDisplay(detected, occluded, reason = "") {
            // 确保获取最新的元素引用
            faceStatusElement = document.getElementById('face-status');

            if (!faceStatusElement) {
                console.error('Face status element not found');
                return;
            }

            // 如果摄像头未启动，强制隐藏状态显示
            if (!cameraActive) {
                faceStatusElement.style.display = 'none';
                faceStatusElement.classList.add('face-status-hidden');
                faceStatusElement.classList.remove('face-status-ok', 'face-status-warning', 'face-status-error');
                return;
            }

            // 确保元素可见
            faceStatusElement.style.display = 'block';

            // 移除所有状态类
            faceStatusElement.classList.remove('face-status-hidden', 'face-status-ok', 'face-status-warning', 'face-status-error');

            const statusText = document.getElementById('face-status-text');

            if (!statusText) {
                console.error('Status text element not found');
                return;
            }

            if (!detected) {
                // 没有检测到人脸
                faceStatusElement.classList.add('face-status-warning');
                statusText.textContent = "<?php echo __('no_face_detected'); ?>";
            } else if (occluded) {
                // 检测到人脸但有遮挡
                faceStatusElement.classList.add('face-status-error');
                statusText.textContent = reason || "<?php echo __('face_occlusion_detected_short'); ?>";
            } else {
                // 检测到人脸且没有遮挡
                faceStatusElement.classList.add('face-status-ok');
                statusText.textContent = "<?php echo __('face_detection_normal'); ?>";
            }
        }

        // 更新进度条
        function updateProgressUI() {
            const userDirExists = document.getElementById('user_dir_exists').value === '1';

            // 获取用户目录状态元素和已有图片数量
            const userDirStatus = document.getElementById('user-dir-status');
            let existingImageCount = 0;

            // 从数据属性中获取图片数量
            if (userDirStatus.dataset.imageCount) {
                existingImageCount = parseInt(userDirStatus.dataset.imageCount, 10);
            }

            console.log(`更新UI - 用户目录存在: ${userDirExists}, 已有图片: ${existingImageCount}, 新拍摄图片: ${capturedImages.length}`);

            // 确定所需照片数量
            let requiredPhotos = REQUIRED_PHOTOS; // 默认需要10张

            // 如果用户目录已存在
            if (userDirExists) {
                // 如果已有10张或以上照片，只需要1张新照片
                if (existingImageCount >= 10) {
                    requiredPhotos = 1;
                } else {
                    // 否则需要凑够10张照片
                    requiredPhotos = Math.max(1, 10 - existingImageCount);
                }
            }

            // 更新所需照片数量显示
            document.getElementById('required-count').textContent = requiredPhotos;

            // 更新提示文本
            const requirementText = document.getElementById('photo-requirement-text');
            if (userDirExists) {
                // 如果用户目录已存在且有图片
                if (existingImageCount > 0) {
                    // 如果已有足够多的图片（例如25张以上），显示至少需要1张新照片
                    if (existingImageCount >= 25) {
                        const remainingSpace = 25 - existingImageCount;
                        if (remainingSpace <= 0) {
                            requirementText.textContent = "<?php echo __('user_photos_limit_reached'); ?>".replace('{count}', existingImageCount);
                        } else if (remainingSpace == 1) {
                            requirementText.textContent = "<?php echo __('user_can_add_only_one_more'); ?>".replace('{count}', existingImageCount);
                        } else {
                            requirementText.textContent = "<?php echo __('user_can_add_more_photos'); ?>".replace('{count}', existingImageCount).replace('{remaining}', remainingSpace);
                        }
                    } else {
                        // 否则显示还需要多少张才能达到25张
                        const neededPhotos = 25 - existingImageCount;
                        requirementText.textContent = "<?php echo __('user_needs_more_photos'); ?>".replace('{count}', existingImageCount).replace('{needed}', neededPhotos);
                    }
                } else {
                    requirementText.textContent = "<?php echo __('user_dir_exists_no_photos'); ?>";
                }
            } else {
                // 获取翻译文本并在JavaScript中替换占位符
                const template = "<?php echo __('new_user_requires_photos'); ?>";
                requirementText.textContent = template.replace('{count}', REQUIRED_PHOTOS);
            }

            // 更新总进度条 - 根据实际需要的照片数量计算
            const totalProgress = Math.min(100, (capturedImages.length / requiredPhotos) * 100);
            document.getElementById('capture-progress').style.width = `${totalProgress}%`;

            // 更新计数
            capturedCount.textContent = capturedImages.length;

            // 计算总数
            const totalCount = existingImageCount + capturedImages.length;

            // 处理通知逻辑
            // 如果总数低于25，重置标记，允许再次显示通知
            if (totalCount < 25) {
                limitNotificationShown = false;
            }
            // 注意：通知现在由updateUserDirectoryStatus函数处理，这里不再显示通知
            // 这样可以避免重复显示通知

            // 更新隐藏的总数计算显示（用于兼容现有代码）
            const totalCountContainer = document.getElementById('total-count-container');
            const existingCountElement = document.getElementById('existing-count');
            const newCountElement = document.getElementById('new-count');
            const totalCountElement = document.getElementById('total-count');

            // 更新新的可见计数器显示
            const existingCountDisplay = document.getElementById('existing-count-display');
            const newCountDisplay = document.getElementById('new-count-display');
            const totalCountDisplay = document.getElementById('total-count-display');

            // 更新两个计数器的值
            if (existingCountDisplay) existingCountDisplay.textContent = existingImageCount;
            if (newCountDisplay) newCountDisplay.textContent = capturedImages.length;
            if (totalCountDisplay) totalCountDisplay.textContent = `${totalCount} / 25`;

            // 根据总数设置颜色
            let countColor = '#4CAF50'; // 默认绿色
            if (totalCount > 25) {
                countColor = '#C51A4A'; // 红色
            } else if (totalCount >= 20) {
                countColor = '#FF9800'; // 橙色
            }

            // 应用颜色到两个计数器
            if (totalCountDisplay) totalCountDisplay.style.color = countColor;

            // 更新隐藏的计数器（用于兼容现有代码）
            if (userDirExists && existingImageCount > 0) {
                // 显示总数计算
                totalCountContainer.style.display = 'none'; // 现在隐藏，因为我们有了新的可见计数器
                existingCountElement.textContent = existingImageCount;
                newCountElement.textContent = capturedImages.length;
                totalCountElement.textContent = totalCount;

                // 设置颜色
                totalCountElement.style.color = countColor;
            } else {
                // 隐藏总数计算
                totalCountContainer.style.display = 'none';
            }

            // 确定是否启用保存按钮
            let enableSaveButton = false;

            // 如果用户目录不存在，需要至少10张照片
            if (!userDirExists && capturedImages.length >= REQUIRED_PHOTOS) {
                enableSaveButton = true;
            }
            // 如果用户目录存在且已有10张或以上照片，只需要1张新照片
            else if (userDirExists && existingImageCount >= 10 && capturedImages.length >= 1) {
                enableSaveButton = true;
            }
            // 如果用户目录存在但照片少于10张，需要凑够10张
            else if (userDirExists && existingImageCount < 10 && (existingImageCount + capturedImages.length) >= 10) {
                enableSaveButton = true;
            }

            saveButton.disabled = !enableSaveButton;

            // 如果达到25张图片的限制，禁用所有按钮
            if (userDirExists && existingImageCount >= 25) {
                captureButton.disabled = true;
                saveButton.disabled = true;
                startButton.disabled = true;
                startButton.title = "<?php echo __('user_image_limit_reached'); ?>";
            }

            // 如果当前拍摄的照片数量达到25张，禁用拍照按钮（减轻树莓派负担）
            else if (capturedImages.length >= 25) {
                captureButton.disabled = true;
                captureButton.title = "<?php echo __('upload_limit_tooltip'); ?>";
            }
            // 如果照片数量少于25张，且摄像头已启动，启用拍照按钮
            else if (cameraActive && video.srcObject) {
                captureButton.disabled = false;
                captureButton.title = "";
            }
        }

        // 声明全局变量用于跟踪人脸面积是否足够
        let faceAreaTooSmall = true;

        // 拍照按钮
        captureButton.addEventListener('click', () => {
            if (!cameraActive) {
                showNotificationAlert("<?php echo __('please_start_camera'); ?>", 'warning');
                return;
            }

            if (!video.srcObject || !faceDetected || faceOccluded || faceAreaTooSmall) {
                // 如果有遮挡，显示提示
                if (faceOccluded) {
                    showNotificationAlert("<?php echo __('face_occlusion_detected'); ?>", 'warning');
                } else if (faceAreaTooSmall) {
                    showNotificationAlert("<?php echo __('face_area_too_small'); ?>", 'warning');
                } else if (!faceDetected) {
                    showNotificationAlert("<?php echo __('no_face_detected'); ?>", 'warning');
                }
                return;
            }

            // 限制一次性拍摄的图片数量为最多25张（减轻树莓派负担）
            if (capturedImages.length >= 25) {
                showNotificationAlert("<?php echo __('capture_limit_exceeded'); ?>", 'warning');
                return;
            }

            // 创建canvas并以人脸框为中心进行1:1裁剪
            const canvas = document.createElement('canvas');

            // 获取当前检测到的人脸框信息
            if (!lastDetectionResults || !lastDetectionResults.detections || lastDetectionResults.detections.length === 0) {
                showNotificationAlert("<?php echo __('cannot_get_face_position'); ?>");
                return;
            }

            // 使用第一个检测到的人脸
            const detection = lastDetectionResults.detections[0];
            const boundingBox = detection.boundingBox;

            // 计算人脸中心点在原始视频中的位置
            const centerX = boundingBox.xCenter * video.videoWidth;
            const centerY = boundingBox.yCenter * video.videoHeight;

            // 计算正方形边长 - 使用与人脸框相同的计算方法，但增加更多边距
            const origWidth = boundingBox.width * video.videoWidth;
            const origHeight = boundingBox.height * video.videoHeight;
            const squareSize = Math.max(origWidth, origHeight) * 1.5; // 增加50%的边距，确保人脸周围有足够空间

            // 确保正方形不会超出视频边界
            const halfSize = squareSize / 2;
            const cropX = Math.max(0, Math.min(video.videoWidth - squareSize, centerX - halfSize));
            const cropY = Math.max(0, Math.min(video.videoHeight - squareSize, centerY - halfSize));

            // 设置canvas为固定大小的正方形 (750x750)
            const FIXED_SIZE = 750; // 固定输出尺寸为750x750像素
            canvas.width = FIXED_SIZE;
            canvas.height = FIXED_SIZE;

            // 绘制裁剪后的图像，并调整为固定大小
            const ctx = canvas.getContext('2d');
            ctx.drawImage(
                video,
                cropX, cropY, squareSize, squareSize, // 源矩形 - 以人脸为中心的正方形区域
                0, 0, FIXED_SIZE, FIXED_SIZE // 目标矩形 - 固定大小的canvas
            );

            // 可选：添加平滑处理，提高图像质量
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';

            // 获取图像数据 - 使用高质量设置
            const imageData = canvas.toDataURL('image/jpeg', 0.95);

            // 验证图像数据
            if (!imageData || imageData.length < 100 || !imageData.startsWith('data:image/jpeg;base64,')) {
                console.error('Invalid image data generated');
                showNotificationAlert('<?php echo __('failed_to_generate_image_data'); ?>');
                return;
            }

            console.log('Image data length:', imageData.length);

            // 添加到预览区域
            addImagePreview(imageData);

            // 更新总计数
            capturedImages.push(imageData);

            // 更新UI
            updateProgressUI();

            // 计算总数 - 通知逻辑已移至 updateProgressUI 函数中，避免重复显示
            const userDirStatus = document.getElementById('user-dir-status');
            const existingImageCount = parseInt(userDirStatus.dataset.imageCount || 0);
            const totalCount = existingImageCount + capturedImages.length;

            // 更新隐藏字段的值
            document.getElementById('image_data').value = imageData;
            document.getElementById('all_images').value = JSON.stringify(capturedImages);

            console.log(`拍照后 - 已有照片: ${userDirStatus.dataset.imageCount || 0}, 新拍照片: ${capturedImages.length}, 总数: ${(parseInt(userDirStatus.dataset.imageCount || 0) + capturedImages.length)}`);
        });

        // 页面加载完成后初始化图片预览模态框
        document.addEventListener('DOMContentLoaded', function() {
            // 图片预览模态框相关变量
            const modal = document.getElementById('image-preview-modal');
            const modalImage = document.getElementById('modal-image');
            const closeModal = document.querySelector('.close-modal');
            const prevButton = document.getElementById('prev-image');
            const nextButton = document.getElementById('next-image');
            let currentImageIndex = 0;

            // 打开模态框并显示图片
            window.openImageModal = function(imageData, index) {
                modalImage.src = imageData;
                currentImageIndex = index;
                modal.style.display = 'flex';
                setTimeout(() => {
                    modal.style.opacity = '1';
                }, 10);

                // 更新导航按钮状态
                updateNavigationButtons();

                // 阻止页面滚动
                document.body.style.overflow = 'hidden';
            };

            // 关闭模态框
            window.closeImageModal = function() {
                modal.style.opacity = '0';
                setTimeout(() => {
                    modal.style.display = 'none';
                }, 300);

                // 恢复页面滚动
                document.body.style.overflow = '';
            };

            // 显示上一张图片
            function showPreviousImage() {
                if (currentImageIndex > 0) {
                    currentImageIndex--;
                    modalImage.src = capturedImages[currentImageIndex];
                    updateNavigationButtons();
                }
            }

            // 显示下一张图片
            function showNextImage() {
                if (currentImageIndex < capturedImages.length - 1) {
                    currentImageIndex++;
                    modalImage.src = capturedImages[currentImageIndex];
                    updateNavigationButtons();
                }
            }

            // 更新导航按钮状态
            function updateNavigationButtons() {
                prevButton.disabled = currentImageIndex === 0;
                prevButton.style.opacity = currentImageIndex === 0 ? '0.3' : '1';

                nextButton.disabled = currentImageIndex === capturedImages.length - 1;
                nextButton.style.opacity = currentImageIndex === capturedImages.length - 1 ? '0.3' : '1';
            }

            // 删除当前显示的图片
            function deleteCurrentImage() {
                if (currentImageIndex >= 0 && currentImageIndex < capturedImages.length) {
                    // 显示确认对话框
                    if (typeof showNotification === 'function') {
                        // 创建确认对话框
                        const confirmDialog = document.createElement('div');
                        confirmDialog.className = 'notification-confirm';
                        confirmDialog.innerHTML = `
                            <div class="notification-confirm-content">
                                <p><?php echo __('confirm_delete_image'); ?></p>
                                <div class="notification-confirm-buttons">
                                    <button class="btn-cancel"><?php echo __('cancel'); ?></button>
                                    <button class="btn-confirm"><?php echo __('delete'); ?></button>
                                </div>
                            </div>
                        `;

                        document.body.appendChild(confirmDialog);

                        // 添加按钮事件
                        const cancelBtn = confirmDialog.querySelector('.btn-cancel');
                        const confirmBtn = confirmDialog.querySelector('.btn-confirm');

                        cancelBtn.addEventListener('click', () => {
                            document.body.removeChild(confirmDialog);
                        });

                        confirmBtn.addEventListener('click', () => {
                            document.body.removeChild(confirmDialog);
                            performDelete();
                        });
                    } else {
                        // 如果通知系统不可用，使用原生确认
                        if (confirm("<?php echo __('confirm_delete_image'); ?>")) {
                            performDelete();
                        }
                    }
                }

                // 执行删除操作
                function performDelete() {
                    // 找到对应的预览项
                    const previewItems = document.querySelectorAll('.preview-item');
                    if (currentImageIndex < previewItems.length) {
                        const previewToDelete = previewItems[currentImageIndex];

                        // 从数组和DOM中移除
                        capturedImages.splice(currentImageIndex, 1);
                        previewContainer.removeChild(previewToDelete);

                        // 更新UI和隐藏字段
                        updateProgressUI();
                        document.getElementById('all_images').value = JSON.stringify(capturedImages);

                        // 获取用户目录状态 - 通知逻辑已移至updateProgressUI函数中
                        const userDirStatus = document.getElementById('user-dir-status');
                        console.log(`删除照片后 - 已有照片: ${userDirStatus.dataset.imageCount || 0}, 新拍照片: ${capturedImages.length}, 总数: ${(parseInt(userDirStatus.dataset.imageCount || 0) + capturedImages.length)}`);

                        // 显示成功通知
                        if (typeof showSuccessNotification === 'function') {
                            showSuccessNotification("<?php echo __('image_deleted_successfully'); ?>");
                        }

                        // 如果还有图像，选择第一个
                        if (capturedImages.length > 0) {
                            // 如果删除的是最后一张，显示前一张
                            if (currentImageIndex >= capturedImages.length) {
                                currentImageIndex = capturedImages.length - 1;
                            }

                            // 更新模态框图片
                            modalImage.src = capturedImages[currentImageIndex];
                            updateNavigationButtons();

                            // 更新选中的图片
                            document.getElementById('image_data').value = capturedImages[currentImageIndex];

                            // 更新预览项的选中状态
                            document.querySelectorAll('.preview-item').forEach((item, idx) => {
                                item.style.border = idx === currentImageIndex ? '3px solid #C51A4A' : '1px solid #ddd';
                            });
                        } else {
                            // 如果没有图像了，关闭模态框
                            document.getElementById('image_data').value = '';
                            window.closeImageModal();
                        }
                    }
                }
            }

            // 绑定模态框事件
            closeModal.addEventListener('click', window.closeImageModal);
            prevButton.addEventListener('click', showPreviousImage);
            nextButton.addEventListener('click', showNextImage);

            // 绑定删除按钮事件
            const deleteButton = document.getElementById('delete-modal-image');
            deleteButton.addEventListener('click', deleteCurrentImage);

            // 点击模态框背景关闭
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    window.closeImageModal();
                }
            });

            // 键盘导航
            document.addEventListener('keydown', (e) => {
                if (modal.style.display !== 'none' && modal.style.display !== '') {
                    if (e.key === 'Escape') {
                        window.closeImageModal();
                    } else if (e.key === 'ArrowLeft') {
                        showPreviousImage();
                    } else if (e.key === 'ArrowRight') {
                        showNextImage();
                    } else if (e.key === 'Delete') {
                        // 添加删除键支持
                        deleteCurrentImage();
                    }
                }
            });
        });

        // 添加图像预览
        function addImagePreview(imageData) {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';

            const img = document.createElement('img');
            img.src = imageData;
            previewItem.appendChild(img);

            // 添加删除按钮
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-preview';
            deleteBtn.innerHTML = '<i class="fas fa-trash"></i>';
            deleteBtn.onclick = function(e) {
                e.stopPropagation(); // 阻止事件冒泡

                // 从数组和DOM中移除
                const index = Array.from(previewContainer.children).indexOf(previewItem);
                if (index >= 0) {
                    capturedImages.splice(index, 1);
                    previewContainer.removeChild(previewItem);

                    // 更新UI和隐藏字段
                    updateProgressUI();
                    document.getElementById('all_images').value = JSON.stringify(capturedImages);

                    const userDirStatus = document.getElementById('user-dir-status');
                    console.log(`删除照片后 - 已有照片: ${userDirStatus.dataset.imageCount || 0}, 新拍照片: ${capturedImages.length}, 总数: ${(parseInt(userDirStatus.dataset.imageCount || 0) + capturedImages.length)}`);

                    // 如果还有图像，选择第一个
                    if (capturedImages.length > 0) {
                        document.getElementById('image_data').value = capturedImages[0];
                    } else {
                        document.getElementById('image_data').value = '';
                    }
                }
            };
            previewItem.appendChild(deleteBtn);

            // 添加点击事件，选择此图像作为要保存的图像
            previewItem.addEventListener('click', (e) => {
                // 如果点击的是删除按钮，不执行以下操作
                if (e.target.closest('.delete-preview')) return;

                // 移除其他预览项的选中状态
                document.querySelectorAll('.preview-item').forEach(item => {
                    item.style.border = '1px solid #ddd';
                });

                // 设置当前项为选中状态
                previewItem.style.border = '3px solid #C51A4A';

                // 更新隐藏字段的值
                const index = Array.from(previewContainer.children).indexOf(previewItem);
                if (index >= 0 && index < capturedImages.length) {
                    // 选中当前图像作为要保存的图像
                    document.getElementById('image_data').value = capturedImages[index];
                    // 确保所有图像数据都被发送
                    document.getElementById('all_images').value = JSON.stringify(capturedImages);

                    // 打开模态框预览图片
                    if (typeof window.openImageModal === 'function') {
                        window.openImageModal(capturedImages[index], index);
                    }
                }
            });

            previewContainer.appendChild(previewItem);

            // 自动选择最新的图像，但不触发模态框
            // 手动设置选中状态
            document.querySelectorAll('.preview-item').forEach(item => {
                item.style.border = '1px solid #ddd';
            });
            previewItem.style.border = '3px solid #C51A4A';

            // 更新隐藏字段的值
            const index = Array.from(previewContainer.children).indexOf(previewItem);
            if (index >= 0 && index < capturedImages.length) {
                document.getElementById('image_data').value = capturedImages[index];
                document.getElementById('all_images').value = JSON.stringify(capturedImages);
            }
        }

        // 当管理员选择不同用户时更新目录状态
        <?php if ($is_admin): ?>
        document.getElementById('fullname').addEventListener('change', function() {
            // 获取选择的用户全名
            const selectedUser = this.value;

            // 保存选择到localStorage，以便页面刷新后恢复
            localStorage.setItem('selectedUser', selectedUser);

            // 清空已捕获的图像
            // 清空预览区域
            previewContainer.innerHTML = '';

            // 重置捕获的图像数组
            capturedImages = [];

            // 重置通知标记
            limitNotificationShown = false;

            // 更新UI
            updateProgressUI();

            // 清空隐藏字段
            document.getElementById('image_data').value = '';
            document.getElementById('all_images').value = '';

            console.log('User changed, cleared captured images');

            // 使用我们的函数检查用户目录状态
            checkUserDirectoryStatus(selectedUser);
        });
        <?php endif; ?>

        // 表单验证
        function validateForm() {
            const fullname = document.getElementById('fullname').value.trim();
            const imageData = document.getElementById('image_data').value;
            const userDirExists = document.getElementById('user_dir_exists').value === '1';

            // 获取用户目录状态元素和已有图片数量
            const userDirStatus = document.getElementById('user-dir-status');
            let existingImageCount = 0;

            // 从数据属性中获取图片数量
            if (userDirStatus.dataset.imageCount) {
                existingImageCount = parseInt(userDirStatus.dataset.imageCount, 10);
            }

            console.log(`验证表单 - 用户目录存在: ${userDirExists}, 已有图片: ${existingImageCount}, 新拍摄图片: ${capturedImages.length}`);

            if (!fullname) {
                showNotificationAlert('<?php echo __('please_enter_name'); ?>', 'warning');
                return false;
            }

            if (!imageData) {
                showNotificationAlert('<?php echo __('please_capture_photo_first'); ?>', 'warning');
                return false;
            }

            // 对于所有用户，至少需要1张照片
            if (capturedImages.length < 1) {
                showNotificationAlert('<?php echo __('please_capture_at_least_one_photo'); ?>', 'warning');
                return false;
            }

            // 限制一次性上传的图片数量为最多25张（减轻树莓派负担）
            if (capturedImages.length > 25) {
                showNotificationAlert("<?php echo __('upload_limit_exceeded'); ?>", 'warning');
                return false;
            }

            // 检查是否会超过最大限制（25张）
            if (userDirExists && (existingImageCount + capturedImages.length) > 25) {
                const allowedNew = 25 - existingImageCount;
                if (allowedNew <= 0) {
                    // 使用不同的消息，避免与红色通知重复
                    showNotificationAlert("<?php echo __('user_can_add_up_to_x'); ?>".replace('{count}', existingImageCount).replace('{allowed}', 0), 'warning');
                } else {
                    showNotificationAlert("<?php echo __('user_can_add_up_to_x'); ?>".replace('{count}', existingImageCount).replace('{allowed}', allowedNew), 'warning');
                }
                return false;
            }

            // 对于新用户，需要至少10张照片
            if (!userDirExists && capturedImages.length < REQUIRED_PHOTOS) {
                // 获取翻译文本并在JavaScript中替换占位符
                const template = "<?php echo __('new_user_requires_photos'); ?>";
                showNotificationAlert(template.replace('{count}', REQUIRED_PHOTOS), 'warning');
                return false;
            }

            // 如果用户目录存在且已有10张或以上照片，只需要1张新照片
            else if (userDirExists && existingImageCount >= 10 && capturedImages.length < 1) {
                showNotificationAlert("<?php echo __('existing_user_requires_one_photo'); ?>", 'warning');
                return false;
            }

            // 如果用户目录存在但图片少于10张，需要凑够10张
            else if (userDirExists && existingImageCount < 10) {
                const totalImages = existingImageCount + capturedImages.length;
                if (totalImages < 10) {
                    const neededPhotos = 10 - existingImageCount;
                    showNotificationAlert("<?php echo __('user_needs_more_photos_to_reach_25'); ?>".replace('{existing}', existingImageCount).replace('{needed}', neededPhotos), 'warning');
                    return false;
                }
            }

            // 保存用户选择到localStorage，确保表单提交后仍然保持选择的用户
            localStorage.setItem('selectedUser', fullname);
            return true;
        }



        // 添加全局键盘事件监听器（隐藏功能）
        document.addEventListener('keydown', function(e) {
            // 检查是否按下空格键，且不是在输入框中
            if (e.code === 'Space' &&
                document.activeElement.tagName !== 'INPUT' &&
                document.activeElement.tagName !== 'TEXTAREA' &&
                document.activeElement.tagName !== 'SELECT' &&
                !document.activeElement.isContentEditable) {

                // 检查拍照按钮是否可用
                if (cameraActive && !captureButton.disabled) {
                    // 模拟点击拍照按钮
                    captureButton.click();

                    // 防止页面滚动
                    e.preventDefault();
                }
            }
        });

        // 页面关闭或刷新前停止摄像头和 MediaPipe
        window.addEventListener('beforeunload', () => {
            if (camera) {
                camera.stop();
            }
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        });

        // 处理图片上传功能
        const uploadButton = document.getElementById('upload-button');
        const imageUploadInput = document.getElementById('image-upload-input');

        // 处理上传按钮点击事件
        uploadButton.addEventListener('click', () => {
            // 检查是否达到图片数量限制
            if (imageCountLimitReached) {
                showNotificationAlert("<?php echo __('user_image_limit_reached_25'); ?>", 'error');
                return;
            }

            // 限制一次性上传的图片数量为最多25张（减轻树莓派负担）
            if (capturedImages.length >= 25) {
                showNotificationAlert("<?php echo __('capture_limit_exceeded'); ?>", 'warning');
                return;
            }

            // 触发文件选择对话框
            imageUploadInput.click();
        });

        // 处理文件选择事件
        imageUploadInput.addEventListener('change', async (event) => {
            const files = event.target.files;

            if (!files || files.length === 0) {
                return;
            }

            // 检查是否达到图片数量限制
            if (imageCountLimitReached) {
                showNotificationAlert("<?php echo __('user_image_limit_reached_25'); ?>", 'error');
                imageUploadInput.value = ''; // 清空文件选择
                return;
            }

            // 限制一次性上传的图片数量为最多25张（减轻树莓派负担）
            if (files.length > 25) {
                showNotificationAlert("<?php echo __('upload_limit_exceeded'); ?>", 'warning');
                imageUploadInput.value = ''; // 清空文件选择
                return;
            }

            // 检查当前已有的图片数量加上新上传的图片数量是否超过25张
            if (capturedImages.length + files.length > 25) {
                showNotificationAlert("<?php echo __('upload_limit_exceeded'); ?>", 'warning');
                imageUploadInput.value = ''; // 清空文件选择
                return;
            }

            // 显示加载中通知
            showNotificationAlert("<?php echo __('processing_images'); ?>", 'info');

            // 创建临时canvas用于处理图片
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const FIXED_SIZE = 750; // 固定输出尺寸为750x750像素
            canvas.width = FIXED_SIZE;
            canvas.height = FIXED_SIZE;

            // 初始化人脸检测器（如果尚未初始化）
            if (!faceDetector) {
                try {
                    await initFaceDetector();
                } catch (error) {
                    console.error('Error initializing face detector for uploads:', error);
                    showNotificationAlert("<?php echo __('error_initializing_face_detector'); ?>", 'error');
                    imageUploadInput.value = ''; // 清空文件选择
                    return;
                }
            }

            // 处理每个文件
            for (let i = 0; i < files.length; i++) {
                const file = files[i];

                // 检查文件类型
                if (!file.type.match('image.*')) {
                    console.error('Invalid file type:', file.type);
                    continue;
                }

                try {
                    // 读取文件为DataURL
                    const imageDataUrl = await readFileAsDataURL(file);

                    // 创建图像对象
                    const img = new Image();
                    img.src = imageDataUrl;

                    // 等待图像加载
                    await new Promise(resolve => {
                        img.onload = resolve;
                    });

                    // 创建临时canvas用于人脸检测
                    const tempCanvas = document.createElement('canvas');
                    const tempCtx = tempCanvas.getContext('2d');

                    // 计算放大后的尺寸 - 确保图片足够大以便更好地检测人脸
                    // 如果原图较小，放大到至少1000px宽度；如果已经足够大，则保持原尺寸
                    const MIN_DETECTION_WIDTH = 1000; // 最小检测宽度
                    let detectionWidth, detectionHeight;

                    if (img.width < MIN_DETECTION_WIDTH) {
                        // 计算放大比例
                        const scale = MIN_DETECTION_WIDTH / img.width;
                        detectionWidth = MIN_DETECTION_WIDTH;
                        detectionHeight = Math.round(img.height * scale);
                        console.log(`放大图片进行检测: ${img.width}x${img.height} -> ${detectionWidth}x${detectionHeight}`);
                    } else {
                        // 图片已经足够大，使用原始尺寸
                        detectionWidth = img.width;
                        detectionHeight = img.height;
                        console.log(`使用原始尺寸进行检测: ${detectionWidth}x${detectionHeight}`);
                    }

                    // 设置canvas尺寸并绘制放大后的图像
                    tempCanvas.width = detectionWidth;
                    tempCanvas.height = detectionHeight;
                    tempCtx.drawImage(img, 0, 0, detectionWidth, detectionHeight);

                    // 获取图像数据
                    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);

                    // 检测人脸
                    const detectionResults = await detectFaceInImage(imageData);

                    if (!detectionResults || !detectionResults.detections || detectionResults.detections.length === 0) {
                        console.error('No face detected in image:', file.name);
                        continue;
                    }

                    // 使用第一个检测到的人脸
                    const detection = detectionResults.detections[0];
                    const boundingBox = detection.boundingBox;

                    // 计算人脸面积占图像的百分比
                    const faceArea = boundingBox.width * boundingBox.height;
                    const totalArea = 1.0; // 归一化坐标系中总面积为1
                    const areaPercentage = (faceArea / totalArea) * 100;

                    // 记录检测到的人脸信息
                    console.log(`检测到人脸 - 文件: ${file.name}, 面积百分比: ${areaPercentage.toFixed(2)}%, 置信度: ${detection.score ? detection.score[0].toFixed(2) : 'N/A'}`);

                    // 检查人脸面积是否足够大（至少15%）- 放宽标准，因为我们已经放大了图像
                    const MIN_FACE_AREA_PERCENT = 15;
                    if (areaPercentage < MIN_FACE_AREA_PERCENT) {
                        console.error(`人脸面积太小: ${file.name}, 面积: ${areaPercentage.toFixed(2)}%, 最小要求: ${MIN_FACE_AREA_PERCENT}%`);
                        continue;
                    }

                    // 计算人脸中心点在原始图像中的位置
                    // 注意：如果我们放大了图像进行检测，需要将坐标映射回原始图像
                    let centerX, centerY, origWidth, origHeight;

                    if (img.width < MIN_DETECTION_WIDTH) {
                        // 计算缩放比例（从检测尺寸到原始尺寸）
                        const scaleBack = img.width / detectionWidth;

                        // 将检测到的坐标映射回原始图像
                        centerX = boundingBox.xCenter * detectionWidth * scaleBack;
                        centerY = boundingBox.yCenter * detectionHeight * scaleBack;

                        // 计算原始图像中的人脸尺寸
                        origWidth = boundingBox.width * detectionWidth * scaleBack;
                        origHeight = boundingBox.height * detectionHeight * scaleBack;

                        console.log(`将坐标映射回原始图像 - 缩放比例: ${scaleBack.toFixed(3)}`);
                    } else {
                        // 直接使用检测结果，无需缩放
                        centerX = boundingBox.xCenter * img.width;
                        centerY = boundingBox.yCenter * img.height;
                        origWidth = boundingBox.width * img.width;
                        origHeight = boundingBox.height * img.height;
                    }

                    // 计算正方形边长 - 使用与人脸框相同的计算方法，但增加更多边距
                    const squareSize = Math.max(origWidth, origHeight) * 1.5; // 增加50%的边距，确保人脸周围有足够空间

                    console.log(`裁剪信息 - 中心点: (${centerX.toFixed(0)}, ${centerY.toFixed(0)}), 人脸尺寸: ${origWidth.toFixed(0)}x${origHeight.toFixed(0)}, 裁剪尺寸: ${squareSize.toFixed(0)}`);

                    // 确保正方形不会超出图像边界
                    const halfSize = squareSize / 2;
                    const cropX = Math.max(0, Math.min(img.width - squareSize, centerX - halfSize));
                    const cropY = Math.max(0, Math.min(img.height - squareSize, centerY - halfSize));

                    // 绘制裁剪后的图像，并调整为固定大小
                    ctx.clearRect(0, 0, FIXED_SIZE, FIXED_SIZE);
                    ctx.drawImage(
                        img,
                        cropX, cropY, squareSize, squareSize, // 源矩形 - 以人脸为中心的正方形区域
                        0, 0, FIXED_SIZE, FIXED_SIZE // 目标矩形 - 固定大小的canvas
                    );

                    // 可选：添加平滑处理，提高图像质量
                    ctx.imageSmoothingEnabled = true;
                    ctx.imageSmoothingQuality = 'high';

                    // 获取图像数据 - 使用高质量设置
                    const processedImageData = canvas.toDataURL('image/jpeg', 0.95);

                    // 验证图像数据
                    if (!processedImageData || processedImageData.length < 100 || !processedImageData.startsWith('data:image/jpeg;base64,')) {
                        console.error('Invalid image data generated for:', file.name);
                        continue;
                    }

                    // 添加到预览区域
                    addImagePreview(processedImageData);

                    // 更新总计数
                    capturedImages.push(processedImageData);

                    // 更新UI
                    updateProgressUI();

                    // 更新隐藏字段的值
                    document.getElementById('image_data').value = processedImageData;
                    document.getElementById('all_images').value = JSON.stringify(capturedImages);

                } catch (error) {
                    console.error('Error processing image:', file.name, error);
                }
            }

            // 清空文件选择，以便可以再次选择相同的文件
            imageUploadInput.value = '';

            // 显示处理完成通知
            if (capturedImages.length > 0) {
                showNotificationAlert("<?php echo __('images_processed_successfully'); ?>", 'success');
            } else {
                showNotificationAlert("<?php echo __('no_valid_faces_found'); ?>", 'warning');
            }
        });

        // 辅助函数：将文件读取为DataURL
        function readFileAsDataURL(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = (e) => resolve(e.target.result);
                reader.onerror = (e) => reject(e);
                reader.readAsDataURL(file);
            });
        }

        // 辅助函数：在图像中检测人脸
        async function detectFaceInImage(imageData) {
            console.log(`开始人脸检测 - 图像尺寸: ${imageData.width}x${imageData.height}`);
            const startTime = performance.now();

            try {
                // 创建一个新的FaceDetection实例，避免与视频流检测冲突
                const detector = new FaceDetection({
                    locateFile: (file) => {
                        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
                    }
                });

                // 设置检测选项
                detector.setOptions({
                    model: 'short', // 使用轻量级模型，速度更快
                    minDetectionConfidence: 0.80 // 稍微降低置信度阈值，提高检测率
                });

                console.log('正在初始化人脸检测器...');
                // 等待加载完成
                await detector.initialize();

                // 使用Promise包装检测过程
                const results = await new Promise((resolve, reject) => {
                    detector.onResults((results) => {
                        resolve(results);
                        // 检测完成后销毁检测器
                        detector.close();
                    });

                    console.log('发送图像数据进行检测...');
                    // 发送图像数据进行检测
                    detector.send({image: imageData}).catch(reject);
                });

                const endTime = performance.now();
                const detectionTime = (endTime - startTime).toFixed(0);

                // 记录检测结果
                if (results.detections && results.detections.length > 0) {
                    console.log(`检测成功 - 找到 ${results.detections.length} 个人脸，耗时: ${detectionTime}ms`);

                    // 记录每个人脸的详细信息
                    results.detections.forEach((detection, index) => {
                        const confidence = detection.score ? detection.score[0] : 'N/A';
                        console.log(`人脸 #${index+1} - 置信度: ${typeof confidence === 'number' ? confidence.toFixed(2) : confidence}`);
                    });
                } else {
                    console.log(`未检测到人脸，耗时: ${detectionTime}ms`);
                }

                return results;
            } catch (error) {
                const endTime = performance.now();
                const detectionTime = (endTime - startTime).toFixed(0);
                console.error(`人脸检测出错，耗时: ${detectionTime}ms, 错误:`, error);
                return null;
            }
        }
    </script>

    <!-- 图片预览模态框 -->
    <div id="image-preview-modal" class="image-preview-modal">
        <div class="modal-content">
            <span class="close-modal">&times;</span>
            <div class="modal-image-container">
                <img id="modal-image" src="" alt="Preview">
            </div>
            <div class="modal-controls">
                <button id="prev-image" class="modal-nav-button"><i class="fas fa-chevron-left"></i></button>
                <button id="delete-modal-image" class="modal-delete-button"><i class="fas fa-trash"></i></button>
                <button id="next-image" class="modal-nav-button"><i class="fas fa-chevron-right"></i></button>
            </div>
        </div>
    </div>
</body>
</html>
