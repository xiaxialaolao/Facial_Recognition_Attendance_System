<?php
session_start();
include 'config.php';
include 'includes/language-loader.php';
include 'includes/notification-functions.php';
include 'includes/log-functions.php';

// 未登录重定向
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    header("Location: login.php");
    exit();
}

// 检查权限 - 只有管理员可以编辑用户
if ($_SESSION['role'] !== 'admin') {
    header("Location: dashboard.php");
    exit();
}

// 检查是否有employee_id参数
if (!isset($_GET['id']) || empty($_GET['id'])) {
    header("Location: dashboard.php");
    exit();
}

$employee_id = $_GET['id'];
$error = "";
$success = "";

// 获取用户信息
$sql = "SELECT employee_id, username, fullname, role, profile_picture FROM users WHERE employee_id = ?";
$stmt = $conn->prepare($sql);
$stmt->bind_param("i", $employee_id);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows !== 1) {
    header("Location: dashboard.php");
    exit();
}

$user = $result->fetch_assoc();

// 处理表单提交
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // 获取表单数据
    $username = $_POST['username'];
    $fullname = $_POST['fullname'];
    $role = $_POST['role'];
    $current_password = $_POST['current_password'] ?? '';
    $new_password = $_POST['new_password'] ?? '';
    $confirm_password = $_POST['confirm_password'] ?? '';

    // 验证数据
    if (empty($username) || empty($fullname) || empty($role)) {
        $error = "All fields are required";
        set_error_notification($error);
        // 不重定向，让错误显示在当前页面
    } else {
        // 开始事务
        $conn->begin_transaction();

        try {
            // 检查是否要更改密码
            if (!empty($current_password) && !empty($new_password)) {
                // 验证当前密码
                $check_password_sql = "SELECT password FROM users WHERE employee_id = ?";
                $check_stmt = $conn->prepare($check_password_sql);
                $check_stmt->bind_param("i", $employee_id);
                $check_stmt->execute();
                $password_result = $check_stmt->get_result();
                $password_row = $password_result->fetch_assoc();

                if (!password_verify($current_password, $password_row['password'])) {
                    throw new Exception("Current password is incorrect");
                }

                if ($new_password !== $confirm_password) {
                    throw new Exception("New password and confirmation do not match");
                }

                // 更新用户信息包括密码
                $hashed_password = password_hash($new_password, PASSWORD_DEFAULT);
                $update_sql = "UPDATE users SET username = ?, fullname = ?, role = ?, password = ? WHERE employee_id = ?";
                $update_stmt = $conn->prepare($update_sql);
                $update_stmt->bind_param("ssssi", $username, $fullname, $role, $hashed_password, $employee_id);
            } else {
                // 只更新用户信息，不更新密码
                $update_sql = "UPDATE users SET username = ?, fullname = ?, role = ? WHERE employee_id = ?";
                $update_stmt = $conn->prepare($update_sql);
                $update_stmt->bind_param("sssi", $username, $fullname, $role, $employee_id);
            }

            // 检查是否有裁剪后的图片数据
            if (!empty($_POST['cropped_image'])) {
                // 从base64字符串中提取图片数据
                $cropped_image = $_POST['cropped_image'];
                $image_parts = explode(";base64,", $cropped_image);

                // 确保数据格式正确
                if (count($image_parts) === 2) {
                    $image_data_base64 = $image_parts[1];
                    $image_data = base64_decode($image_data_base64);

                    // 创建文件名 - 使用用户的全名
                    $filename = $fullname . '.jpg';
                    // 只替换文件名中的特殊字符，保留空格
                    $filename = preg_replace('/[^a-zA-Z0-9_\-\. ]/', '', $filename);
                    $profile_pictures_dir = '/home/xiaxialaolao/FRAS_env/Profile_Pictures';
                    $filepath = $profile_pictures_dir . '/' . $filename;

                    // 确保目录存在
                    if (!file_exists($profile_pictures_dir)) {
                        mkdir($profile_pictures_dir, 0755, true);
                    }

                    // 检查并删除旧的图片文件（如果存在）
                    if (file_exists($filepath)) {
                        unlink($filepath);
                    }

                    // 保存图片到文件系统
                    $result = file_put_contents($filepath, $image_data);

                    if ($result !== false) {
                        // 确保文件权限正确
                        chmod($filepath, 0644);

                        // 强制文件系统同步，确保更改立即可见
                        clearstatcache(true, $filepath);
                    }

                    // 同时更新数据库中的头像数据
                    $update_image_sql = "UPDATE users SET profile_picture = ? WHERE employee_id = ?";
                    $update_image_stmt = $conn->prepare($update_image_sql);
                    $update_image_stmt->bind_param("si", $image_data, $employee_id);
                    $update_image_stmt->execute();
                } else {
                    throw new Exception("Invalid image data format");
                }
            }

            // 获取原始用户数据进行比较
            $original_sql = "SELECT username, fullname, role FROM users WHERE employee_id = ?";
            $original_stmt = $conn->prepare($original_sql);
            $original_stmt->bind_param("i", $employee_id);
            $original_stmt->execute();
            $original_result = $original_stmt->get_result();
            $original_user = $original_result->fetch_assoc();

            // 构建更改日志消息
            $changes = array();

            // 检查哪些字段被更改
            if ($original_user['username'] != $username) {
                $changes[] = "Username: {$original_user['username']} -> $username";
            }
            if ($original_user['fullname'] != $fullname) {
                $changes[] = "Full name: {$original_user['fullname']} -> $fullname";
            }
            if ($original_user['role'] != $role) {
                $changes[] = "Role: {$original_user['role']} -> $role";
            }

            // 检查是否更改了密码
            if (!empty($current_password) && !empty($new_password)) {
                $changes[] = "Password changed";
            }

            // 检查是否更新了头像
            if (!empty($_POST['cropped_image'])) {
                $changes[] = "Avatar updated";
            }

            // 执行更新
            $update_stmt->execute();

            // 记录用户更新日志
            if (count($changes) > 0) {
                $log_message = "User information updated: $username (" . implode(", ", $changes) . ")";
            } else {
                $log_message = "User information updated: $username (No changes)";
            }
            log_info($log_message, "edit.php");

            // 提交事务
            $conn->commit();

            // 设置成功通知
            set_success_notification(__('user_updated'));

            // 重定向到同一页面，避免表单重复提交和通知重复显示
            header("Location: edit.php?id=" . $employee_id);
            exit();

        } catch (Exception $e) {
            // 回滚事务
            $conn->rollback();

            $error = $e->getMessage();
            set_error_notification($error);
            // 不重定向，让错误显示在当前页面
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
    <link rel="stylesheet" href="css/edit-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.css">
    <?php
    include 'includes/theme-loader.php';
    include 'includes/date-formatter.php';
    include 'includes/notifications-loader.php';
    ?>
    <title><?php echo __('edit_user_information'); ?> - FRAS System</title>
</head>
<body>
<div class="dashboard-container">
        <?php include 'includes/sidebar.php'; ?>

        <!-- 主内容区 -->
        <div class="main-content">
            <!-- 顶部标题区 -->
            <div class="content-header">
                <h2 class="content-title"><?php echo __('edit_user_information'); ?></h2>

                <button onclick="window.location.href='dashboard.php'" class="action-button">
                    <i class="fas fa-arrow-left"></i>
                    <?php echo __('back_to_dashboard'); ?>
                </button>
            </div>

            <!-- 表单区域 -->
            <div class="form-container" style="max-width: 800px; margin-left: auto; margin-right: auto;">
                <!-- 通知由 notifications-loader.php 处理 -->

                <form method="POST" enctype="multipart/form-data">
                    <div class="form-group">
                        <label><?php echo __('employee_id'); ?>:</label>
                        <input type="text" value="<?php echo htmlspecialchars($user['employee_id']); ?>" disabled>
                    </div>

                    <div class="form-group">
                        <label><?php echo __('current_avatar'); ?>:</label>
                        <img src="data:image/jpeg;base64,<?php echo base64_encode($user['profile_picture']); ?>" alt="Profile Picture" class="profile-preview">
                    </div>

                    <div class="form-group">
                        <label for="profile_picture"><?php echo __('update_avatar'); ?>:</label>
                        <input type="hidden" name="cropped_image" id="cropped_image">
                        <label for="profile_picture_input" class="file-input-label">
                            <i class="fas fa-upload"></i> <?php echo __('upload'); ?>
                        </label>
                        <input type="file" id="profile_picture_input" class="file-input" accept="image/*">
                        <span id="selected-file-name" style="margin-left: 10px; font-size: 14px;"></span>
                    </div>

                    <div class="form-group">
                        <label for="username"><?php echo __('username'); ?>:</label>
                        <input type="text" id="username" name="username" value="<?php echo htmlspecialchars($user['username']); ?>" required>
                    </div>

                    <div class="form-group">
                        <label for="fullname"><?php echo __('full_name'); ?>:</label>
                        <input type="text" id="fullname" name="fullname" value="<?php echo htmlspecialchars($user['fullname']); ?>" required>
                    </div>

                    <div class="form-group">
                        <label for="role"><?php echo __('role'); ?>:</label>
                        <select id="role" name="role" required>
                            <option value="user" <?php echo ($user['role'] == 'user') ? 'selected' : ''; ?>><?php echo __('user'); ?></option>
                            <option value="admin" <?php echo ($user['role'] == 'admin') ? 'selected' : ''; ?>><?php echo __('admin'); ?></option>
                        </select>
                    </div>

                    <div class="password-section">
                        <h3><?php echo __('change_password'); ?></h3>

                        <div class="form-group">
                            <label for="current_password"><?php echo __('current_password'); ?>:</label>
                            <input type="password" id="current_password" name="current_password">
                        </div>

                        <div class="form-group">
                            <label for="new_password"><?php echo __('new_password'); ?>:</label>
                            <input type="password" id="new_password" name="new_password">
                        </div>

                        <div class="form-group">
                            <label for="confirm_password"><?php echo __('confirm_password'); ?>:</label>
                            <input type="password" id="confirm_password" name="confirm_password">
                        </div>
                    </div>

                    <div class="button-group">
                        <a href="dashboard.php" class="cancel-button">
                            <i class="fas fa-times"></i> <?php echo __('cancel'); ?>
                        </a>
                        <button type="submit" class="submit-button">
                            <i class="fas fa-save"></i> <?php echo __('save_changes'); ?>
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- Image Cropper Modal -->
    <div class="image-cropper-container" id="image-cropper-modal">
        <div class="cropper-box">
            <div class="cropper-title"><?php echo __('crop_image'); ?></div>
            <div class="img-container">
                <img id="image-to-crop" src="" alt="Image to crop">
            </div>
            <div class="img-preview"></div>
            <div class="cropper-buttons">
                <button type="button" class="crop-cancel" id="crop-cancel"><?php echo __('cancel'); ?></button>
                <button type="button" class="crop-submit" id="crop-submit"><?php echo __('apply'); ?></button>
            </div>
        </div>
    </div>

    <!-- JavaScript Libraries -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.12/cropper.min.js"></script>

    <script>
        // Image cropper functionality
        $(document).ready(function() {
            var $imageToCrop = $('#image-to-crop');
            var $cropperModal = $('#image-cropper-modal');
            var $fileInput = $('#profile_picture_input');
            var $croppedImageInput = $('#cropped_image');
            var $selectedFileName = $('#selected-file-name');
            var cropper;

            // When file is selected
            $fileInput.on('change', function(e) {
                var files = e.target.files;
                var fileName = files[0].name;
                $selectedFileName.text(fileName);

                if (files && files.length > 0) {
                    var reader = new FileReader();
                    reader.onload = function(e) {
                        $imageToCrop.attr('src', e.target.result);
                        $cropperModal.css('display', 'block');

                        // Initialize cropper
                        if (cropper) {
                            cropper.destroy();
                        }

                        cropper = new Cropper($imageToCrop[0], {
                            aspectRatio: 1,
                            viewMode: 1,
                            dragMode: 'move',
                            autoCropArea: 1,
                            restore: false,
                            guides: true,
                            center: true,
                            highlight: false,
                            cropBoxMovable: true,
                            cropBoxResizable: true,
                            toggleDragModeOnDblclick: false,
                            preview: '.img-preview'
                        });
                    };
                    reader.readAsDataURL(files[0]);
                }
            });

            // Cancel cropping
            $('#crop-cancel').on('click', function() {
                if (cropper) {
                    cropper.destroy();
                    cropper = null;
                }
                $cropperModal.css('display', 'none');
                $fileInput.val('');
                $selectedFileName.text('');
            });

            // Apply cropping
            $('#crop-submit').on('click', function() {
                if (cropper) {
                    var canvas = cropper.getCroppedCanvas({
                        width: 300,
                        height: 300,
                        minWidth: 100,
                        minHeight: 100,
                        maxWidth: 1000,
                        maxHeight: 1000,
                        fillColor: '#fff',
                        imageSmoothingEnabled: true,
                        imageSmoothingQuality: 'high'
                    });

                    if (canvas) {
                        // Convert canvas to base64 string
                        var croppedImageData = canvas.toDataURL('image/jpeg');
                        $croppedImageInput.val(croppedImageData);

                        // Show preview of cropped image
                        $('.profile-preview').attr('src', croppedImageData);

                        // Close modal
                        $cropperModal.css('display', 'none');
                        cropper.destroy();
                        cropper = null;
                    }
                }
            });
        });
    </script>
</body>
</html>
