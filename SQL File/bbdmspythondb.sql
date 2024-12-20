-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Aug 17, 2024 at 02:32 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `bbdmspythondb`
--

-- --------------------------------------------------------

--
-- Table structure for table `auth_group`
--

CREATE TABLE `auth_group` (
  `id` int(11) NOT NULL,
  `name` varchar(150) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `auth_group_permissions`
--

CREATE TABLE `auth_group_permissions` (
  `id` bigint(20) NOT NULL,
  `group_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `auth_permission`
--

CREATE TABLE `auth_permission` (
  `id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `content_type_id` int(11) NOT NULL,
  `codename` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `auth_permission`
--

INSERT INTO `auth_permission` (`id`, `name`, `content_type_id`, `codename`) VALUES
(1, 'Can add log entry', 1, 'add_logentry'),
(2, 'Can change log entry', 1, 'change_logentry'),
(3, 'Can delete log entry', 1, 'delete_logentry'),
(4, 'Can view log entry', 1, 'view_logentry'),
(5, 'Can add permission', 2, 'add_permission'),
(6, 'Can change permission', 2, 'change_permission'),
(7, 'Can delete permission', 2, 'delete_permission'),
(8, 'Can view permission', 2, 'view_permission'),
(9, 'Can add group', 3, 'add_group'),
(10, 'Can change group', 3, 'change_group'),
(11, 'Can delete group', 3, 'delete_group'),
(12, 'Can view group', 3, 'view_group'),
(13, 'Can add content type', 4, 'add_contenttype'),
(14, 'Can change content type', 4, 'change_contenttype'),
(15, 'Can delete content type', 4, 'delete_contenttype'),
(16, 'Can view content type', 4, 'view_contenttype'),
(17, 'Can add session', 5, 'add_session'),
(18, 'Can change session', 5, 'change_session'),
(19, 'Can delete session', 5, 'delete_session'),
(20, 'Can view session', 5, 'view_session'),
(21, 'Can add user', 6, 'add_customuser'),
(22, 'Can change user', 6, 'change_customuser'),
(23, 'Can delete user', 6, 'delete_customuser'),
(24, 'Can view user', 6, 'view_customuser'),
(25, 'Can add bloodgroup', 7, 'add_bloodgroup'),
(26, 'Can change bloodgroup', 7, 'change_bloodgroup'),
(27, 'Can delete bloodgroup', 7, 'delete_bloodgroup'),
(28, 'Can view bloodgroup', 7, 'view_bloodgroup'),
(29, 'Can add donor reg', 8, 'add_donorreg'),
(30, 'Can change donor reg', 8, 'change_donorreg'),
(31, 'Can delete donor reg', 8, 'delete_donorreg'),
(32, 'Can view donor reg', 8, 'view_donorreg'),
(33, 'Can add blood request', 9, 'add_bloodrequest'),
(34, 'Can change blood request', 9, 'change_bloodrequest'),
(35, 'Can delete blood request', 9, 'delete_bloodrequest'),
(36, 'Can view blood request', 9, 'view_bloodrequest'),
(37, 'Can add contact', 10, 'add_contact'),
(38, 'Can change contact', 10, 'change_contact'),
(39, 'Can delete contact', 10, 'delete_contact'),
(40, 'Can view contact', 10, 'view_contact');

-- --------------------------------------------------------

--
-- Table structure for table `bbdmsapp_bloodgroup`
--

CREATE TABLE `bbdmsapp_bloodgroup` (
  `id` bigint(20) NOT NULL,
  `bloodgroup` varchar(200) NOT NULL,
  `created_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `bbdmsapp_bloodgroup`
--

INSERT INTO `bbdmsapp_bloodgroup` (`id`, `bloodgroup`, `created_at`, `updated_at`) VALUES
(1, 'O+', '2024-07-12 13:30:20.442050', '2024-07-12 13:30:20.442050'),
(3, 'AB+', '2024-07-12 13:31:06.897807', '2024-07-12 13:31:06.897807'),
(4, 'O-', '2024-07-12 13:31:14.761853', '2024-07-18 05:27:12.840156'),
(5, 'A+', '2024-07-12 13:31:46.727622', '2024-07-12 13:31:46.727622'),
(7, 'B+', '2024-07-12 13:31:57.739773', '2024-07-12 13:31:57.739773'),
(8, 'B-', '2024-07-12 13:32:02.377069', '2024-07-12 13:32:02.377069'),
(9, 'AB-', '2024-07-12 13:33:53.159584', '2024-07-12 13:33:53.159584');

-- --------------------------------------------------------

--
-- Table structure for table `bbdmsapp_bloodrequest`
--

CREATE TABLE `bbdmsapp_bloodrequest` (
  `id` bigint(20) NOT NULL,
  `fullname` varchar(250) NOT NULL,
  `mobno` varchar(11) NOT NULL,
  `email` varchar(250) NOT NULL,
  `requirer` varchar(250) NOT NULL,
  `message` longtext NOT NULL,
  `regdate_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL,
  `donid_id` bigint(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `bbdmsapp_bloodrequest`
--

INSERT INTO `bbdmsapp_bloodrequest` (`id`, `fullname`, `mobno`, `email`, `requirer`, `message`, `regdate_at`, `updated_at`, `donid_id`) VALUES
(1, 'Rakesh Sharma', '7979797987', 'rakesh@gmail.com', 'Father', 'Please Help!!', '2024-07-19 06:29:19.233586', '2024-07-19 06:29:19.233586', 1),
(5, 'Test', '7479879797', 'test@gmail.com', 'wife', 'ghgjhjh', '2024-07-19 07:00:33.229193', '2024-07-19 07:00:33.229193', 3),
(6, 'Test', '7479879797', 'test@gmail.com', 'wife', 'ghgjhjh', '2024-07-19 07:01:01.112982', '2024-07-19 07:01:01.112982', 3),
(7, 'Ram', '7987897987', 'ram@gmail.com', 'wife', 'Please Help!!', '2024-07-19 07:02:38.686591', '2024-07-19 07:02:38.686591', 5),
(9, 'Rahul', '4561231230', 'rahul12@test.com', 'Father', 'Required urgent blood for my father', '2024-08-06 16:56:05.415685', '2024-08-06 16:56:05.415685', 8),
(10, 'Karan Saxena', '6546546546', 'karan@gmail.com', 'Father', 'Plz Help', '2024-08-08 12:20:28.866821', '2024-08-08 12:20:28.866821', 9),
(11, 'Amit Kumar', '1231231230', 'amitk12@test.com', 'Father', 'My Father need the blood uregntly', '2024-08-15 16:25:45.330415', '2024-08-15 16:25:45.330415', 10);

-- --------------------------------------------------------

--
-- Table structure for table `bbdmsapp_contact`
--

CREATE TABLE `bbdmsapp_contact` (
  `id` bigint(20) NOT NULL,
  `fullname` varchar(250) NOT NULL,
  `mobno` varchar(11) NOT NULL,
  `email` varchar(250) NOT NULL,
  `message` longtext NOT NULL,
  `status` varchar(50) NOT NULL,
  `regdate_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `bbdmsapp_contact`
--

INSERT INTO `bbdmsapp_contact` (`id`, `fullname`, `mobno`, `email`, `message`, `status`, `regdate_at`, `updated_at`) VALUES
(1, 'Megha Singh', '4561237899', 'megha@gmail.com', 'rytrtyrtyrty', 'Read', '2024-07-19 11:40:24.922719', '2024-07-22 11:54:32.552710'),
(2, 'Test', '9798798798', 'test@gmail.com', 'Please Help!!', 'Read', '2024-07-19 11:43:17.687546', '2024-08-06 17:03:17.082474'),
(3, 'Rakesh Sharma', '9798798798', 'rakesh@gmail.com', 'Need B+ blood', 'Read', '2024-07-22 04:28:00.292038', '2024-08-06 17:03:19.858638');

-- --------------------------------------------------------

--
-- Table structure for table `bbdmsapp_customuser`
--

CREATE TABLE `bbdmsapp_customuser` (
  `id` bigint(20) NOT NULL,
  `password` varchar(128) NOT NULL,
  `last_login` datetime(6) DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) NOT NULL,
  `first_name` varchar(150) NOT NULL,
  `last_name` varchar(150) NOT NULL,
  `email` varchar(254) NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  `user_type` varchar(50) NOT NULL,
  `profile_pic` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `bbdmsapp_customuser`
--

INSERT INTO `bbdmsapp_customuser` (`id`, `password`, `last_login`, `is_superuser`, `username`, `first_name`, `last_name`, `email`, `is_staff`, `is_active`, `date_joined`, `user_type`, `profile_pic`) VALUES
(1, 'pbkdf2_sha256$390000$eJbJHEbAlPqKmoe4iiFiTM$ZeRWKiPvCbpHHXHyJuh2N52tENO31wy7LzgPCAYhQUY=', '2024-08-17 12:15:37.309486', 1, 'admin', 'Admin', 'Test', 'admin@gmail.com', 1, 1, '2024-07-12 06:48:39.434307', '1', 'media/profile_pic/img.jpg'),
(2, 'pbkdf2_sha256$720000$RClT1aCbMbZTMc5xlCFpg5$N0Wv09v4rK4qzIibVkVoY72jdrFzBvkAttcVJZ0wpWI=', '2024-07-22 04:31:35.284748', 0, 'abir123', 'Abir', 'Singh', 'abir@gmail.com', 0, 1, '2024-07-18 06:13:53.109087', '2', 'media/profile_pic/img_Fa9m1II.jpg'),
(3, 'pbkdf2_sha256$390000$bTTYQ7ULWeRhFe0nNNYUUJ$9A16PzTk0cQYc3Os6Gti6Ezgzs0VCFYjXraB1wZQKbg=', '2024-08-17 12:28:00.213846', 0, 'test123', 'Test', 'Sample', 'test@gmail.com', 0, 1, '2024-07-18 06:18:35.270108', '2', 'media/profile_pic/img_VZCJn0m.jpg'),
(4, 'pbkdf2_sha256$720000$x2Qa1KGgwqsIzPkF3E2gM7$P6D8VRpsCdj4b+VE4khPFda3EJl2U0WYI5Da0zK8O8I=', '2024-07-22 05:10:43.336065', 0, 'rahul123', 'Rahul', 'Singh', 'rahul@gmail.com', 0, 1, '2024-07-18 06:22:06.229272', '2', 'media/profile_pic/test-thumb3.jpg'),
(5, 'pbkdf2_sha256$720000$9VftrfwrBnXKlWptoguH0o$PlKv0+8nxLlnrAyuBIXa8hwMvxWi+JcUJZ6GECCQNm0=', '2024-07-23 06:59:08.162076', 0, 'meenu123', 'Meenu', 'Kumari', 'meenu@gmail.com', 0, 1, '2024-07-19 06:51:47.503529', '2', 'media/profile_pic/img_ZU2IBc2.jpg'),
(6, 'pbkdf2_sha256$720000$YlUIao1B56X6K1f3Uk8vqO$H9f+fTiaij97rRdAACcq9AyCMLeBNwp2hcNIFVRo7SY=', '2024-07-23 04:53:09.080266', 0, 'anuj123', 'Anuj', 'Kumar', 'anuj@gmail.com', 0, 1, '2024-07-19 06:55:32.743366', '2', 'media/profile_pic/8074804_pUNAmZ2.png'),
(7, 'pbkdf2_sha256$720000$83CVgEXyuy2my8ov68Q7vu$k7PBaYs9rrGq92TfhDnvQvWuh5bvsGEv1sDChg16NWw=', NULL, 0, 'test@1234', 'Test', 'Test', 'test123@gmail.com', 0, 1, '2024-07-19 06:57:20.160168', '2', 'media/profile_pic/blog-8.jpg'),
(9, 'pbkdf2_sha256$720000$iKRf9NkmV71kST2EHhHTn5$jKGAbjs8qp7vZ8uBn78q+JSH0XduNyxUTp10mY/tmtk=', '2024-07-23 05:00:01.308715', 0, 'ramesh123', 'Ramesh', ' Jha', 'ramesh@gmail.com', 0, 1, '2024-07-23 04:59:47.852910', '2', 'media/profile_pic/hero-img.png'),
(10, 'pbkdf2_sha256$390000$eJbJHEbAlPqKmoe4iiFiTM$ZeRWKiPvCbpHHXHyJuh2N52tENO31wy7LzgPCAYhQUY=', '2024-08-06 16:54:23.707275', 0, 'ak30', 'Anuj', 'Kumar', 'ak123@gmail.com', 0, 1, '2024-08-06 16:54:15.911423', '2', 'media/profile_pic/doctor_1.png'),
(11, 'pbkdf2_sha256$390000$71NcWfYCduv6DOs7aH8jUK$mUQQqiNZyDctolO+OlNNgUKPpPRQOUILC3efY51Rnyo=', '2024-08-08 12:15:52.327534', 0, 'sat123', 'Satyendra', 'Kumar', 'satyendra@gmail.com', 0, 1, '2024-08-08 12:14:57.195684', '2', 'media/profile_pic/man.png'),
(12, 'pbkdf2_sha256$390000$wnZTNRltISrHL789UvqfQ9$c9GBJsXQVQ6JrrcL1OWWP0CI+PkDTBUgD4/xD6GPbTg=', '2024-08-15 16:24:36.887792', 0, 'john123', 'John ', 'Doe', 'jhn12@test.com', 0, 1, '2024-08-15 16:24:20.705817', '2', 'media/profile_pic/man_iA0r8l1.png');

-- --------------------------------------------------------

--
-- Table structure for table `bbdmsapp_customuser_groups`
--

CREATE TABLE `bbdmsapp_customuser_groups` (
  `id` bigint(20) NOT NULL,
  `customuser_id` bigint(20) NOT NULL,
  `group_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `bbdmsapp_customuser_user_permissions`
--

CREATE TABLE `bbdmsapp_customuser_user_permissions` (
  `id` bigint(20) NOT NULL,
  `customuser_id` bigint(20) NOT NULL,
  `permission_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `bbdmsapp_donorreg`
--

CREATE TABLE `bbdmsapp_donorreg` (
  `id` bigint(20) NOT NULL,
  `age` int(11) NOT NULL,
  `mobilenumber` varchar(11) NOT NULL,
  `gender` varchar(100) NOT NULL,
  `address` varchar(250) NOT NULL,
  `regdate_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL,
  `admin_id` bigint(20) DEFAULT NULL,
  `bloodgroup_id` bigint(20) NOT NULL,
  `status` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `bbdmsapp_donorreg`
--

INSERT INTO `bbdmsapp_donorreg` (`id`, `age`, `mobilenumber`, `gender`, `address`, `regdate_at`, `updated_at`, `admin_id`, `bloodgroup_id`, `status`) VALUES
(1, 36, '4987464789', 'Female', 'O-908, GHU, Block-7 Lucknow, Uttar Pradesh', '2024-07-18 06:18:36.037832', '2024-08-08 13:31:57.622553', 3, 3, 'Hide'),
(3, 26, '9879797798', 'Female', 'O-908, GHU, Block-7 Laxmi Nagar Delhi', '2024-07-19 06:51:48.293523', '2024-07-23 11:27:23.481118', 5, 4, '0'),
(5, 25, '1465456456', 'Male', 'O-908, GHU, Block-7', '2024-07-19 06:57:20.949568', '2024-07-19 06:57:20.949568', 7, 3, '0'),
(7, 45, '9746457987', 'Male', '                           \r\n                edfewrfewrtretgryyt        \r\n                            ', '2024-07-23 04:59:48.649736', '2024-07-23 04:59:48.649736', 9, 7, '0'),
(8, 32, '1231231230', 'Male', '  NA', '2024-08-06 16:54:16.195629', '2024-08-06 16:57:26.586653', 10, 7, '0'),
(9, 29, '6465465465', 'Male', '\r\n K-901, Jawahar Nagra New Delhi                           \r\n                        \r\n                           ', '2024-08-08 12:14:57.489676', '2024-08-08 12:17:21.578066', 11, 9, '0'),
(10, 25, '1414253652', 'Male', 'Test Address   \r\n                            ', '2024-08-15 16:24:21.010713', '2024-08-15 16:26:59.219918', 12, 7, 'Hide');

-- --------------------------------------------------------

--
-- Table structure for table `django_admin_log`
--

CREATE TABLE `django_admin_log` (
  `id` int(11) NOT NULL,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext DEFAULT NULL,
  `object_repr` varchar(200) NOT NULL,
  `action_flag` smallint(5) UNSIGNED NOT NULL CHECK (`action_flag` >= 0),
  `change_message` longtext NOT NULL,
  `content_type_id` int(11) DEFAULT NULL,
  `user_id` bigint(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `django_content_type`
--

CREATE TABLE `django_content_type` (
  `id` int(11) NOT NULL,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `django_content_type`
--

INSERT INTO `django_content_type` (`id`, `app_label`, `model`) VALUES
(1, 'admin', 'logentry'),
(3, 'auth', 'group'),
(2, 'auth', 'permission'),
(7, 'bbdmsapp', 'bloodgroup'),
(9, 'bbdmsapp', 'bloodrequest'),
(10, 'bbdmsapp', 'contact'),
(6, 'bbdmsapp', 'customuser'),
(8, 'bbdmsapp', 'donorreg'),
(4, 'contenttypes', 'contenttype'),
(5, 'sessions', 'session');

-- --------------------------------------------------------

--
-- Table structure for table `django_migrations`
--

CREATE TABLE `django_migrations` (
  `id` bigint(20) NOT NULL,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `django_migrations`
--

INSERT INTO `django_migrations` (`id`, `app`, `name`, `applied`) VALUES
(1, 'contenttypes', '0001_initial', '2024-07-12 06:47:14.126703'),
(2, 'contenttypes', '0002_remove_content_type_name', '2024-07-12 06:47:14.198992'),
(3, 'auth', '0001_initial', '2024-07-12 06:47:14.462476'),
(4, 'auth', '0002_alter_permission_name_max_length', '2024-07-12 06:47:14.513112'),
(5, 'auth', '0003_alter_user_email_max_length', '2024-07-12 06:47:14.526079'),
(6, 'auth', '0004_alter_user_username_opts', '2024-07-12 06:47:14.533058'),
(7, 'auth', '0005_alter_user_last_login_null', '2024-07-12 06:47:14.540312'),
(8, 'auth', '0006_require_contenttypes_0002', '2024-07-12 06:47:14.551045'),
(9, 'auth', '0007_alter_validators_add_error_messages', '2024-07-12 06:47:14.571142'),
(10, 'auth', '0008_alter_user_username_max_length', '2024-07-12 06:47:14.584678'),
(11, 'auth', '0009_alter_user_last_name_max_length', '2024-07-12 06:47:14.593685'),
(12, 'auth', '0010_alter_group_name_max_length', '2024-07-12 06:47:14.616442'),
(13, 'auth', '0011_update_proxy_permissions', '2024-07-12 06:47:14.624417'),
(14, 'auth', '0012_alter_user_first_name_max_length', '2024-07-12 06:47:14.645651'),
(15, 'bbdmsapp', '0001_initial', '2024-07-12 06:47:14.899976'),
(16, 'admin', '0001_initial', '2024-07-12 06:47:14.995710'),
(17, 'admin', '0002_logentry_remove_auto_add', '2024-07-12 06:47:15.007680'),
(18, 'admin', '0003_logentry_add_action_flag_choices', '2024-07-12 06:47:15.018947'),
(19, 'bbdmsapp', '0002_alter_customuser_user_type', '2024-07-12 06:47:15.031902'),
(20, 'sessions', '0001_initial', '2024-07-12 06:47:15.067361'),
(21, 'bbdmsapp', '0003_bloodgroup_alter_customuser_user_type', '2024-07-12 13:29:01.051537'),
(22, 'bbdmsapp', '0004_alter_customuser_user_type_donorreg', '2024-07-18 06:09:10.691475'),
(23, 'bbdmsapp', '0005_bloodrequest', '2024-07-19 06:12:29.806453'),
(24, 'bbdmsapp', '0006_contact_alter_customuser_user_type', '2024-07-19 11:39:31.030484'),
(25, 'bbdmsapp', '0007_donorreg_status_alter_customuser_user_type', '2024-07-22 12:54:28.387708');

-- --------------------------------------------------------

--
-- Table structure for table `django_session`
--

CREATE TABLE `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `auth_group`
--
ALTER TABLE `auth_group`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`);

--
-- Indexes for table `auth_group_permissions`
--
ALTER TABLE `auth_group_permissions`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  ADD KEY `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` (`permission_id`);

--
-- Indexes for table `auth_permission`
--
ALTER TABLE `auth_permission`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `auth_permission_content_type_id_codename_01ab375a_uniq` (`content_type_id`,`codename`);

--
-- Indexes for table `bbdmsapp_bloodgroup`
--
ALTER TABLE `bbdmsapp_bloodgroup`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `bbdmsapp_bloodrequest`
--
ALTER TABLE `bbdmsapp_bloodrequest`
  ADD PRIMARY KEY (`id`),
  ADD KEY `bbdmsapp_bloodrequest_donid_id_0bb1a046_fk_bbdmsapp_donorreg_id` (`donid_id`);

--
-- Indexes for table `bbdmsapp_contact`
--
ALTER TABLE `bbdmsapp_contact`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `bbdmsapp_customuser`
--
ALTER TABLE `bbdmsapp_customuser`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`);

--
-- Indexes for table `bbdmsapp_customuser_groups`
--
ALTER TABLE `bbdmsapp_customuser_groups`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `bbdmsapp_customuser_groups_customuser_id_group_id_e9442ea2_uniq` (`customuser_id`,`group_id`),
  ADD KEY `bbdmsapp_customuser_groups_group_id_e05eab4f_fk_auth_group_id` (`group_id`);

--
-- Indexes for table `bbdmsapp_customuser_user_permissions`
--
ALTER TABLE `bbdmsapp_customuser_user_permissions`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `bbdmsapp_customuser_user_customuser_id_permission_6e312c2f_uniq` (`customuser_id`,`permission_id`),
  ADD KEY `bbdmsapp_customuser__permission_id_7d2e242c_fk_auth_perm` (`permission_id`);

--
-- Indexes for table `bbdmsapp_donorreg`
--
ALTER TABLE `bbdmsapp_donorreg`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `admin_id` (`admin_id`),
  ADD KEY `bbdmsapp_donorreg_bloodgroup_id_161d9e3b_fk_bbdmsapp_` (`bloodgroup_id`);

--
-- Indexes for table `django_admin_log`
--
ALTER TABLE `django_admin_log`
  ADD PRIMARY KEY (`id`),
  ADD KEY `django_admin_log_content_type_id_c4bce8eb_fk_django_co` (`content_type_id`),
  ADD KEY `django_admin_log_user_id_c564eba6_fk_bbdmsapp_customuser_id` (`user_id`);

--
-- Indexes for table `django_content_type`
--
ALTER TABLE `django_content_type`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`);

--
-- Indexes for table `django_migrations`
--
ALTER TABLE `django_migrations`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `django_session`
--
ALTER TABLE `django_session`
  ADD PRIMARY KEY (`session_key`),
  ADD KEY `django_session_expire_date_a5c62663` (`expire_date`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `auth_group`
--
ALTER TABLE `auth_group`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `auth_group_permissions`
--
ALTER TABLE `auth_group_permissions`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `auth_permission`
--
ALTER TABLE `auth_permission`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=41;

--
-- AUTO_INCREMENT for table `bbdmsapp_bloodgroup`
--
ALTER TABLE `bbdmsapp_bloodgroup`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=13;

--
-- AUTO_INCREMENT for table `bbdmsapp_bloodrequest`
--
ALTER TABLE `bbdmsapp_bloodrequest`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;

--
-- AUTO_INCREMENT for table `bbdmsapp_contact`
--
ALTER TABLE `bbdmsapp_contact`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT for table `bbdmsapp_customuser`
--
ALTER TABLE `bbdmsapp_customuser`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=13;

--
-- AUTO_INCREMENT for table `bbdmsapp_customuser_groups`
--
ALTER TABLE `bbdmsapp_customuser_groups`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `bbdmsapp_customuser_user_permissions`
--
ALTER TABLE `bbdmsapp_customuser_user_permissions`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `bbdmsapp_donorreg`
--
ALTER TABLE `bbdmsapp_donorreg`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT for table `django_admin_log`
--
ALTER TABLE `django_admin_log`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `django_content_type`
--
ALTER TABLE `django_content_type`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT for table `django_migrations`
--
ALTER TABLE `django_migrations`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=26;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `auth_group_permissions`
--
ALTER TABLE `auth_group_permissions`
  ADD CONSTRAINT `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`),
  ADD CONSTRAINT `auth_group_permissions_group_id_b120cbf9_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`);

--
-- Constraints for table `auth_permission`
--
ALTER TABLE `auth_permission`
  ADD CONSTRAINT `auth_permission_content_type_id_2f476e4b_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`);

--
-- Constraints for table `bbdmsapp_bloodrequest`
--
ALTER TABLE `bbdmsapp_bloodrequest`
  ADD CONSTRAINT `bbdmsapp_bloodrequest_donid_id_0bb1a046_fk_bbdmsapp_donorreg_id` FOREIGN KEY (`donid_id`) REFERENCES `bbdmsapp_donorreg` (`id`);

--
-- Constraints for table `bbdmsapp_customuser_groups`
--
ALTER TABLE `bbdmsapp_customuser_groups`
  ADD CONSTRAINT `bbdmsapp_customuser__customuser_id_b669c5dc_fk_bbdmsapp_` FOREIGN KEY (`customuser_id`) REFERENCES `bbdmsapp_customuser` (`id`),
  ADD CONSTRAINT `bbdmsapp_customuser_groups_group_id_e05eab4f_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`);

--
-- Constraints for table `bbdmsapp_customuser_user_permissions`
--
ALTER TABLE `bbdmsapp_customuser_user_permissions`
  ADD CONSTRAINT `bbdmsapp_customuser__customuser_id_d1454c39_fk_bbdmsapp_` FOREIGN KEY (`customuser_id`) REFERENCES `bbdmsapp_customuser` (`id`),
  ADD CONSTRAINT `bbdmsapp_customuser__permission_id_7d2e242c_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`);

--
-- Constraints for table `bbdmsapp_donorreg`
--
ALTER TABLE `bbdmsapp_donorreg`
  ADD CONSTRAINT `bbdmsapp_donorreg_admin_id_2a2036e0_fk_bbdmsapp_customuser_id` FOREIGN KEY (`admin_id`) REFERENCES `bbdmsapp_customuser` (`id`),
  ADD CONSTRAINT `bbdmsapp_donorreg_bloodgroup_id_161d9e3b_fk_bbdmsapp_` FOREIGN KEY (`bloodgroup_id`) REFERENCES `bbdmsapp_bloodgroup` (`id`);

--
-- Constraints for table `django_admin_log`
--
ALTER TABLE `django_admin_log`
  ADD CONSTRAINT `django_admin_log_content_type_id_c4bce8eb_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`),
  ADD CONSTRAINT `django_admin_log_user_id_c564eba6_fk_bbdmsapp_customuser_id` FOREIGN KEY (`user_id`) REFERENCES `bbdmsapp_customuser` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
